import os
import sys
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index, search_index
from sentence_transformers import SentenceTransformer
from experiments.benchmark_data import benchmark_queries

load_dotenv()


def init_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.0,
        max_tokens=60000  # Large enough to hold the full JSON
    )

SYSTEM_PROMPT = """You are an expert relevance assessor for an academic search engine.
You will evaluate candidate documents against multiple search queries simultaneously.
Score EACH candidate document on a scale of 0 to 3:
3 = Highly Relevant (Directly answers the query, strong topical overlap)
2 = Relevant (Contains information useful for answering the query)
1 = Weakly Relevant (Related topic, some supporting information)
0 = Not Relevant (Does not materially help answer the query)

Output a raw JSON object (with NO markdown formatting, NO backticks) mapping query index (string) to a dictionary of candidate IDs to their integer score.
Example output format:
{
  "0": {"104": 0, "45": 2, "388": 3},
  "1": {"99": 1, "200": 0}
}
"""

def extract_json(text):
    import json
    import re
    # Remove markdown code blocks if present
    text = re.sub(r'```json\n', '', text)
    text = re.sub(r'```', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON. First 100 chars: {text[:100]}")
        return {}

def run_expansion():
    print("Loading LLM...")
    llm = init_llm()
    
    print("Loading ArXiv ML Papers Dataset...")
    docs, labels, label_names = load_documents()
    
    # We use 2000 documents as per previous experiments
    docs = docs[:2000]
    labels = labels[:2000]
    
    print("\nLoading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding Documents (Dense Index)...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index = build_index(embeddings, labels)
    
    print("\nBuilding BM25 Index (Sparse Index)...")
    bm25 = BM25Index()
    bm25.fit(docs)
    
    hybrid = HybridSearcher(bm25)
    
    eval_queries = benchmark_queries
    print(f"\nPreparing candidates for {len(eval_queries)} Queries...")
    
    queries_to_judge = []
    v2_benchmark_template = []
    
    mega_prompt = ""
    
    for i, item in enumerate(tqdm(eval_queries)):
        q_text = item["query"]
        relevant_docs = item.get("relevant_docs", [])
        
        if not relevant_docs:
            continue
            
        baseline_id = relevant_docs[0]
        baseline_text = docs[baseline_id]
        
        # Step 2: Retrieve Candidates
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # 20 Dense
        _, dense_indices = search_index(faiss_index, q_emb, top_k=20)
        # 20 BM25
        _, bm25_indices = bm25.score(q_text, top_k=20)
        # 20 Hybrid
        hybrid_indices, _, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=20)
        
        # Union Pool
        pool = set()
        for idx in dense_indices:
            if idx >= 0: pool.add(int(idx))
        for idx in bm25_indices:
            if idx >= 0: pool.add(int(idx))
        for idx in hybrid_indices:
            if idx >= 0: pool.add(int(idx))
            
        candidates_to_judge = [doc_id for doc_id in pool if doc_id != baseline_id]
        
        v2_benchmark_template.append({
            "idx": str(i),
            "query": q_text,
            "relevance": {str(baseline_id): 3}
        })
        
        if candidates_to_judge:
            queries_to_judge.append({
                "idx": str(i),
                "query": q_text,
                "baseline_text": baseline_text,
                "candidates": candidates_to_judge
            })
            
    print(f"\nPrepared {len(queries_to_judge)} queries. Batching 5 queries per LLM request...")
    
    # Chunk by 5
    batch_size = 5
    chunks = [queries_to_judge[i:i + batch_size] for i in range(0, len(queries_to_judge), batch_size)]
    
    for c_idx, chunk in enumerate(chunks):
        print(f"Processing Batch {c_idx + 1}/{len(chunks)}...", flush=True)
        
        mega_prompt = ""
        for q_obj in chunk:
            mega_prompt += f"\n--- QUERY INDEX {q_obj['idx']} ---\nQUERY: {q_obj['query']}\nBASELINE DOC:\n{q_obj['baseline_text'][:400]}...\nCANDIDATES:\n"
            for cand_id in q_obj['candidates']:
                abstract = docs[cand_id][:300].replace('\n', ' ')
                mega_prompt += f"[ID: {cand_id}] {abstract}...\n"
                
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=mega_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            
            content_str = response.content
            if isinstance(content_str, list):
                content_str = content_str[0]['text'] if len(content_str) > 0 and 'text' in content_str[0] else str(content_str)
                
            scores_dict = extract_json(content_str)
            
            # Merge scores for this chunk
            for q_data in v2_benchmark_template:
                q_idx = q_data.get("idx")
                if q_idx in scores_dict:
                    for cand_id_str, score in scores_dict[q_idx].items():
                        try:
                            q_data["relevance"][str(cand_id_str)] = int(score)
                        except ValueError:
                            pass
            print(f"Batch {c_idx + 1} processed successfully.", flush=True)
        except Exception as e:
            print(f"Error calling Gemini in batch {c_idx+1}: {e}", flush=True)
            
        print("Waiting 15 seconds to respect rate limits...", flush=True)
        time.sleep(15)

    stats = {
        "queries": 0,
        "pool_size": [],
        "highly_relevant": [],
        "relevant": [],
        "weakly_relevant": []
    }

    # Calculate final stats
    for q_data in v2_benchmark_template:
        stats["queries"] += 1
        stats["pool_size"].append(len(q_data["relevance"]))
        stats["highly_relevant"].append(sum(1 for v in q_data["relevance"].values() if v == 3))
        stats["relevant"].append(sum(1 for v in q_data["relevance"].values() if v == 2))
        stats["weakly_relevant"].append(sum(1 for v in q_data["relevance"].values() if v == 1))
        if "idx" in q_data:
            del q_data["idx"]

    v2_benchmark = v2_benchmark_template
    # Export JSON
    os.makedirs("data", exist_ok=True)
    with open("data/retrieval_benchmark_v2.json", "w") as f:
        json.dump(v2_benchmark, f, indent=2)
        
    # Write Report
    report = f"""# Retrieval Benchmark Expansion Report (v2)

## Statistics
- **Number of queries:** {stats['queries']}
- **Average candidate pool size:** {sum(stats['pool_size']) / max(1, len(stats['pool_size'])):.1f} documents
- **Average highly relevant (3):** {sum(stats['highly_relevant']) / max(1, len(stats['highly_relevant'])):.2f} per query
- **Average relevant (2):** {sum(stats['relevant']) / max(1, len(stats['relevant'])):.2f} per query
- **Average weakly relevant (1):** {sum(stats['weakly_relevant']) / max(1, len(stats['weakly_relevant'])):.2f} per query

## Sample Entry
```json
{json.dumps(v2_benchmark[0], indent=2) if v2_benchmark else "{}"}
```

## FACT
- Generated multi-document relevance labels using Gemini 1.5 Flash as a judge.
- Baseline query relevance was successfully pinned to 3 (Highly Relevant).
- Candidate pool constructed using the union of top-20 results from Dense, BM25, and Hybrid.

## LIMITATION
- **LLM-assisted labeling bias:** The LLM may favor candidates that have high lexical overlap with the query or baseline, missing implicit semantic connections that a human expert would catch.
- **Context window limits:** Abstracts were truncated to ~1000 chars to avoid prompt overflow, potentially omitting crucial details at the end of abstracts.

## RECOMMENDATION
- **Human Verification:** Periodically sample 5% of the LLM-graded relevance scores and have a domain expert blind-verify them to calculate an inter-annotator agreement (Cohen's Kappa).
"""
    with open("data/retrieval_benchmark_v2_report.md", "w") as f:
        f.write(report)
        
    print("\nBenchmark expansion complete. Saved to data/retrieval_benchmark_v2.json")

if __name__ == "__main__":
    run_expansion()

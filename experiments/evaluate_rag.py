import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index
from sentence_transformers import SentenceTransformer
from app.llm import generate_answer, GeminiProvider
from experiments.benchmark_data import benchmark_queries

try:
    from app.reranker import get_reranker
except ImportError:
    get_reranker = None

# LLM-as-a-judge prompt templates
FAITHFULNESS_PROMPT = """You are an impartial judge evaluating the faithfulness of an AI-generated answer.
Your task is to determine if all claims made in the ANSWER are directly supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Does the ANSWER contain any claims or information not present in the CONTEXT?
Score 1.0 if the answer is completely faithful to the context (no hallucinations).
Score 0.5 if it is partially faithful but includes some unsupported claims.
Score 0.0 if the answer is entirely hallucinated or contradicts the context.

Respond ONLY with the numerical score (e.g., 1.0, 0.5, 0.0).
"""

RELEVANCE_PROMPT = """You are an impartial judge evaluating how well an AI-generated answer addresses a user's query.

QUERY:
{query}

ANSWER:
{answer}

Does the ANSWER directly address the QUERY?
Score 1.0 if the answer fully and directly addresses the query.
Score 0.5 if the answer is tangential or only partially addresses the query.
Score 0.0 if the answer is completely irrelevant.

Respond ONLY with the numerical score (e.g., 1.0, 0.5, 0.0).
"""

def extract_score(llm_response: str) -> float:
    try:
        # Try to extract the first float found in the response
        import re
        matches = re.findall(r'\d+\.\d+', llm_response)
        if matches:
            return float(matches[0])
        # Try integers
        matches = re.findall(r'\d+', llm_response)
        if matches:
            return float(matches[0])
    except:
        pass
    return 0.0

def run_rag_evaluation():
    print("Loading ArXiv ML Papers Dataset...")
    docs, labels, _ = load_documents()
    docs = docs[:2000]
    labels = labels[:2000]
    
    print("Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding Documents...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index = build_index(embeddings, labels)
    
    print("Building BM25 Index...")
    bm25 = BM25Index()
    bm25.fit(docs)
    hybrid = HybridSearcher(bm25)
    
    reranker = get_reranker() if get_reranker else None
    llm_provider = GeminiProvider()
    
    if not llm_provider.model:
        print("GEMINI_API_KEY not found. Skipping RAG Evaluation.")
        return

    # To avoid rate limits, evaluate on the first 10 queries
    eval_queries = benchmark_queries[:10]
    
    print(f"\nEvaluating RAG Pipeline on {len(eval_queries)} queries...")
    
    results = []
    faithfulness_scores = []
    relevance_scores = []
    
    for i, item in enumerate(eval_queries):
        query = item["query"]
        print(f"\n[{i+1}/{len(eval_queries)}] Query: {query}")
        
        # 1. Retrieve
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        cand_indices, _, _ = hybrid.search(query, q_emb, faiss_index, docs, top_k=30)
        cand_docs = [docs[idx] for idx in cand_indices]
        
        if reranker:
            final_indices, _ = reranker.rerank(query, cand_docs, cand_indices, top_k=3)
        else:
            final_indices = cand_indices[:3]
            
        retrieved_docs = [docs[idx] for idx in final_indices]
        
        # 2. Generate Answer
        print("  -> Generating Answer...")
        answer = generate_answer(query, retrieved_docs, provider=llm_provider)
        
        if not answer:
            print("  -> LLM Generation failed.")
            continue
            
        context_text = "\n\n".join(retrieved_docs)
        
        # 3. Evaluate Faithfulness
        f_prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
        f_response = llm_provider.generate(f_prompt)
        f_score = extract_score(f_response) if f_response else 0.0
        
        # 4. Evaluate Relevance
        r_prompt = RELEVANCE_PROMPT.format(query=query, answer=answer)
        r_response = llm_provider.generate(r_prompt)
        r_score = extract_score(r_response) if r_response else 0.0
        
        print(f"  -> Faithfulness: {f_score} | Relevance: {r_score}")
        
        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)
        
        results.append({
            "query": query,
            "answer": answer,
            "faithfulness": f_score,
            "relevance": r_score
        })
        
        # Sleep to avoid rate limits
        time.sleep(2)
        
    print("\n" + "="*50)
    print("RAG EVALUATION METRICS (LLM-AS-A-JUDGE)")
    print("="*50)
    print(f"Faithfulness:     {np.mean(faithfulness_scores):.4f}")
    print(f"Answer Relevance: {np.mean(relevance_scores):.4f}")
    print("="*50)
    
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/rag_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_rag_evaluation()

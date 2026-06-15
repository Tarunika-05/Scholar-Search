import json
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index
from sentence_transformers import SentenceTransformer

try:
    from app.reranker import get_reranker
except ImportError:
    get_reranker = None

def run_diagnostic():
    print("Loading docs...")
    docs, labels, label_names = load_documents()
    docs = docs[:2000]
    labels = labels[:2000]
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index = build_index(embeddings, labels)
    
    bm25 = BM25Index()
    bm25.fit(docs)
    
    hybrid = HybridSearcher(bm25)
    reranker = get_reranker()
    _ = reranker.model
    
    with open("data/retrieval_benchmark_v2.json", "r") as f:
        benchmark = json.load(f)
        
    # Sample 10 queries that have highly relevant docs in the top 2000
    valid_queries = []
    for item in benchmark:
        relevance = item.get("relevance", {})
        valid_relevance = {int(k): v for k, v in relevance.items() if int(k) < 2000}
        if any(v >= 3 for v in valid_relevance.values()):
            valid_queries.append((item, valid_relevance))
            
    random.seed(42)
    sample = random.sample(valid_queries, min(10, len(valid_queries)))
    
    with open("experiments/results/reranker_diagnostic_report.md", "w") as f:
        f.write("# Reranker Diagnostic Report\n\n")
        
        for i, (item, relevance) in enumerate(sample):
            q_text = item["query"]
            f.write(f"## Query {i+1}: {q_text}\n\n")
            
            # Expected
            expected_docs = [doc_id for doc_id, score in relevance.items() if score >= 3]
            f.write(f"**Expected Highly Relevant (Score >= 3):** {expected_docs}\n\n")
            
            q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            # Hybrid
            hybrid_indices, hybrid_scores, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=100)
            hybrid_indices = [int(idx) for idx in hybrid_indices if idx >= 0]
            
            f.write(f"**Hybrid Top 10:** {hybrid_indices[:10]}\n\n")
            
            # Reranked
            cand_docs = [docs[idx] for idx in hybrid_indices]
            r_indices, r_scores = reranker.rerank(q_text, cand_docs, hybrid_indices, top_k=10)
            
            f.write(f"**Reranked Top 10:** {r_indices[:10]}\n\n")
            
            # Did expected move up or down?
            for exp_id in expected_docs:
                h_rank = hybrid_indices.index(exp_id) + 1 if exp_id in hybrid_indices else ">100"
                r_rank = r_indices.index(exp_id) + 1 if exp_id in r_indices else ">10"
                f.write(f"- Doc {exp_id}: Hybrid Rank {h_rank} -> Reranker Rank {r_rank}\n")
            f.write("\n---\n\n")

if __name__ == "__main__":
    run_diagnostic()

import sys
import os
import numpy as np
import json

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index, search_index
from app.evaluation import precision_at_k, mrr, ndcg_at_k, recall_at_k, average_precision
from sentence_transformers import SentenceTransformer


try:
    from app.reranker import get_reranker
except ImportError:
    get_reranker = None

def run_evaluation():
    print("Loading ArXiv ML Papers Dataset...")
    docs, labels, label_names = load_documents()
    
    # Use 2000 documents for evaluation
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
    
    reranker = None
    if get_reranker is not None:
        try:
            reranker = get_reranker()
            _ = reranker.model  # Force load
        except Exception as e:
            print(f"Failed to load reranker: {e}")
            
    with open("data/retrieval_benchmark_v2.json", "r") as f:
        benchmark_queries = json.load(f)
        
    print(f"\nEvaluating Search Modes using {len(benchmark_queries)} Natural Language Queries from V2 benchmark...")
    
    metrics = {
        "Dense": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []},
        "BM25": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []},
        "Hybrid": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []},
        "Reranked": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []}
    }
    
    cat_to_id = {name: i for i, name in enumerate(label_names)}
    
    for item in benchmark_queries:
        q_text = item["query"]
        relevant_docs_raw = item.get("relevance", {})
        
        # Convert string keys to int and filter out-of-bounds
        relevant_docs = {}
        for doc_id_str, score in relevant_docs_raw.items():
            doc_id = int(doc_id_str)
            if doc_id < len(docs):
                relevant_docs[doc_id] = score
        
        if not relevant_docs:
            continue
            
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Dense Search
        dense_distances, dense_indices = search_index(faiss_index, q_emb, top_k=10)
        dense_indices = [int(idx) for idx in dense_indices if idx >= 0]
        if dense_indices:
            metrics["Dense"]["p@3"].append(precision_at_k(relevant_docs, dense_indices, 3))
            metrics["Dense"]["mrr"].append(mrr(relevant_docs, dense_indices))
            metrics["Dense"]["ndcg@10"].append(ndcg_at_k(relevant_docs, dense_indices, 10))
            metrics["Dense"]["recall@10"].append(recall_at_k(relevant_docs, dense_indices, 10))
            metrics["Dense"]["map"].append(average_precision(relevant_docs, dense_indices))
        
        # BM25 Search
        _, bm25_indices = bm25.score(q_text, top_k=10)
        bm25_indices = [int(idx) for idx in bm25_indices if idx >= 0]
        if bm25_indices:
            metrics["BM25"]["p@3"].append(precision_at_k(relevant_docs, bm25_indices, 3))
            metrics["BM25"]["mrr"].append(mrr(relevant_docs, bm25_indices))
            metrics["BM25"]["ndcg@10"].append(ndcg_at_k(relevant_docs, bm25_indices, 10))
            metrics["BM25"]["recall@10"].append(recall_at_k(relevant_docs, bm25_indices, 10))
            metrics["BM25"]["map"].append(average_precision(relevant_docs, bm25_indices))
        
        # Hybrid Search
        hybrid_indices, _, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=10)
        hybrid_indices = [int(idx) for idx in hybrid_indices if idx >= 0]
        if hybrid_indices:
            metrics["Hybrid"]["p@3"].append(precision_at_k(relevant_docs, hybrid_indices, 3))
            metrics["Hybrid"]["mrr"].append(mrr(relevant_docs, hybrid_indices))
            metrics["Hybrid"]["ndcg@10"].append(ndcg_at_k(relevant_docs, hybrid_indices, 10))
            metrics["Hybrid"]["recall@10"].append(recall_at_k(relevant_docs, hybrid_indices, 10))
            metrics["Hybrid"]["map"].append(average_precision(relevant_docs, hybrid_indices))
            
        # Reranked Hybrid (Top-100 retrieved -> Top-10 reranked with Interpolation)
        if reranker is not None:
            cand_indices, cand_scores, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=100)
            cand_indices = [int(idx) for idx in cand_indices if idx >= 0]
            if cand_indices:
                cand_docs = [docs[idx] for idx in cand_indices]
                r_indices, _ = reranker.rerank(q_text, cand_docs, cand_indices, top_k=10)
                metrics["Reranked"]["p@3"].append(precision_at_k(relevant_docs, r_indices, 3))
                metrics["Reranked"]["mrr"].append(mrr(relevant_docs, r_indices))
                metrics["Reranked"]["ndcg@10"].append(ndcg_at_k(relevant_docs, r_indices, 10))
                metrics["Reranked"]["recall@10"].append(recall_at_k(relevant_docs, r_indices, 10))
                metrics["Reranked"]["map"].append(average_precision(relevant_docs, r_indices))

    print("\n" + "="*70)
    print(f"{'Mode':<12} | {'P@3':<8} | {'MRR':<8} | {'NDCG@10':<8} | {'R@10':<8} | {'MAP':<8}")
    print("-" * 70)
    
    modes = ["BM25", "Dense", "Hybrid"]
    if reranker is not None:
        modes.append("Reranked")
        
    final_metrics = {}
    
    for mode in modes:
        if not metrics[mode]["p@3"]:
            continue
        p3 = np.mean(metrics[mode]["p@3"])
        mrr_val = np.mean(metrics[mode]["mrr"])
        ndcg = np.mean(metrics[mode]["ndcg@10"])
        rec = np.mean(metrics[mode]["recall@10"])
        m_ap = np.mean(metrics[mode]["map"])
        
        final_metrics[mode] = {
            "p@3": float(p3),
            "mrr": float(mrr_val),
            "ndcg@10": float(ndcg),
            "recall@10": float(rec),
            "map": float(m_ap)
        }
        
        print(f"{mode:<12} | {p3:.4f}   | {mrr_val:.4f}   | {ndcg:.4f}   | {rec:.4f}   | {m_ap:.4f}")
    print("="*70)
    
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/evaluation_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    run_evaluation()

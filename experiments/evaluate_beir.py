import sys
import os
import numpy as np
from datasets import load_dataset

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index, search_index
from app.evaluation import precision_at_k, mrr, ndcg_at_k, recall_at_k, average_precision
from sentence_transformers import SentenceTransformer

try:
    from app.reranker import get_reranker
except ImportError:
    get_reranker = None

def run_beir_evaluation():
    print("Loading BEIR SciFact Dataset...")
    
    # Load corpus
    corpus = load_dataset('mteb/scifact', 'corpus', split='corpus')
    docs = []
    doc_id_to_idx = {}
    for i, item in enumerate(corpus):
        doc_text = f"{item['title']} {item['text']}"
        docs.append(doc_text.strip())
        doc_id_to_idx[item['_id']] = i
        
    print(f"Loaded {len(docs)} documents.")
    
    # Load queries
    queries = load_dataset('mteb/scifact', 'queries', split='queries')
    query_id_to_text = {item['_id']: item['text'] for item in queries}
    
    # Load qrels (ground truth)
    qrels = load_dataset('mteb/scifact', 'default', split='test')
    
    # Group qrels by query_id
    from collections import defaultdict
    benchmark_queries = defaultdict(list)
    for item in qrels:
        q_id = item['query-id']
        doc_id = item['corpus-id']
        score = item['score']
        if score > 0 and doc_id in doc_id_to_idx:
            benchmark_queries[q_id].append(doc_id_to_idx[doc_id])
            
    # Format into list
    eval_queries = []
    for q_id, rel_docs in benchmark_queries.items():
        if rel_docs and q_id in query_id_to_text:
            eval_queries.append({
                "query": query_id_to_text[q_id],
                "relevant_docs": rel_docs
            })
            
    eval_queries = eval_queries[:50]  # Back to 50 queries now that we use the fast S-PubMedBert model
    print(f"Loaded {len(eval_queries)} queries with ground truth.")
    
    # Initialize Search Components
    print("\nLoading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding Documents (Dense Index)...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    # Fake labels for building the index
    labels = [0] * len(docs)
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
            
    print(f"\nEvaluating Search Modes using {len(eval_queries)} Natural Language Queries...")
    
    metrics = {
        "Dense": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []},
        "BM25": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []},
        "Hybrid": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []},
        "Reranked": {"p@3": [], "mrr": [], "ndcg@10": [], "recall@10": [], "map": []}
    }
    
    for i, item in enumerate(eval_queries):
        print(f"Evaluating query {i+1}/{len(eval_queries)}...", end="\r")
        q_text = item["query"]
        relevant_docs = item["relevant_docs"]
        
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
            
        # Reranked Hybrid (Top-20 retrieved -> Top-10 reranked with Interpolation)
        if reranker is not None:
            cand_indices, cand_scores, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=20)
            cand_indices = [int(idx) for idx in cand_indices if idx >= 0]
            if cand_indices:
                cand_docs = [docs[idx] for idx in cand_indices]
                r_indices, _ = reranker.rerank(q_text, cand_docs, cand_indices, hybrid_scores=cand_scores, top_k=10)
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
        
    for mode in modes:
        if not metrics[mode]["p@3"]:
            continue
        p3 = np.mean(metrics[mode]["p@3"])
        mrr_val = np.mean(metrics[mode]["mrr"])
        ndcg = np.mean(metrics[mode]["ndcg@10"])
        rec = np.mean(metrics[mode]["recall@10"])
        m_ap = np.mean(metrics[mode]["map"])
        
        print(f"{mode:<12} | {p3:.4f}   | {mrr_val:.4f}   | {ndcg:.4f}   | {rec:.4f}   | {m_ap:.4f}")
    print("="*70)

if __name__ == '__main__':
    run_beir_evaluation()

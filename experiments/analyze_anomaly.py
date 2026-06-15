import sys
import os
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher, load_bm25
from app.vector_store import build_index
from sentence_transformers import SentenceTransformer
from app.evaluation import recall_at_k, average_precision
try:
    from app.reranker import get_reranker
except ImportError:
    pass

def run_analysis():
    print("Loading documents...")
    docs, labels, label_names = load_documents()
    docs = docs[:2000]
    labels = labels[:2000]
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index = build_index(embeddings, labels)
    
    print("Loading BM25...")
    bm25 = load_bm25()
    if bm25 is None:
        bm25 = BM25Index()
        bm25.fit(docs)
    
    hybrid = HybridSearcher(bm25)
    reranker = get_reranker()
    
    with open("data/retrieval_benchmark_v2.json", "r") as f:
        queries = json.load(f)
        
    analysis_results = []
    
    for item in queries:
        q_text = item["query"]
        relevant_docs_raw = item.get("relevance", {})
        
        relevant_docs = {}
        for doc_id_str, score in relevant_docs_raw.items():
            doc_id = int(doc_id_str)
            if doc_id < len(docs):
                relevant_docs[doc_id] = score
                
        if not relevant_docs:
            continue
            
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Hybrid
        h_indices, _, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=10)
        h_indices = [int(idx) for idx in h_indices if idx >= 0]
        h_recall = recall_at_k(relevant_docs, h_indices, 10)
        h_map = average_precision(relevant_docs, h_indices)
        
        # Reranked
        cand_indices, _, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=100)
        cand_indices = [int(idx) for idx in cand_indices if idx >= 0]
        
        if not cand_indices:
            continue
            
        cand_docs = [docs[idx] for idx in cand_indices]
        r_indices, _ = reranker.rerank(q_text, cand_docs, cand_indices, top_k=10)
        r_recall = recall_at_k(relevant_docs, r_indices, 10)
        r_map = average_precision(relevant_docs, r_indices)
        
        # We want to find cases where Hybrid Recall or MAP is strictly better than Reranker
        if h_recall > r_recall or h_map > r_map:
            analysis_results.append({
                "query": q_text,
                "relevant_docs": relevant_docs,
                "hybrid_top10": h_indices,
                "hybrid_recall": h_recall,
                "hybrid_map": h_map,
                "reranker_top10": r_indices,
                "reranker_recall": r_recall,
                "reranker_map": r_map
            })
            
    # Take up to 20 samples
    sample_size = min(20, len(analysis_results))
    samples = random.sample(analysis_results, sample_size)
    
    with open("experiments/reranker_anomaly_analysis.json", "w") as f:
        json.dump(samples, f, indent=2)
        
    print(f"Found {len(analysis_results)} queries where Hybrid beat Reranker. Saved {sample_size} samples.")

if __name__ == "__main__":
    run_analysis()

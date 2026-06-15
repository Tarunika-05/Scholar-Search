import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index
from app.reranker import get_reranker
from sentence_transformers import SentenceTransformer

def debug():
    print("Loading ArXiv dataset...")
    docs, labels, _ = load_documents()
    docs = docs[:2000]
    labels = labels[:2000]
    
    print("Loading model and building indexes...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index = build_index(embeddings, labels)
    
    bm25 = BM25Index()
    bm25.fit(docs)
    
    hybrid = HybridSearcher(bm25)
    reranker = get_reranker()
    _ = reranker.model  # Force lazy load
    
    queries = [
        "What mechanism replaces recurrence in transformers?",
        "How do convolutional neural networks detect edges in images?",
        "What is policy gradient in reinforcement learning?",
        "What are graph neural networks used for?",
        "How does batch normalization stabilize training?"
    ]
    
    for q in queries:
        print("\n" + "="*80)
        print(f"QUERY: {q}")
        print("="*80)
        
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Hybrid Top 10
        h_indices, h_scores, _ = hybrid.search(q, q_emb, faiss_index, docs, top_k=10)
        print("\n--- HYBRID TOP 10 ---")
        for i, (idx, score) in enumerate(zip(h_indices, h_scores)):
            doc_preview = docs[idx][:100].replace('\n', ' ')
            print(f"{i+1}. [Idx: {idx}] (Score: {score:.4f}) {doc_preview}...")
            
        # Reranked Top 10 (From Top 50)
        cand_indices, _, _ = hybrid.search(q, q_emb, faiss_index, docs, top_k=50)
        cand_docs = [docs[idx] for idx in cand_indices]
        r_indices, r_scores = reranker.rerank(q, cand_docs, cand_indices, top_k=10)
        
        print("\n--- RERANKED TOP 10 (From 50 Candidates) ---")
        for i, (idx, score) in enumerate(zip(r_indices, r_scores)):
            doc_preview = docs[idx][:100].replace('\n', ' ')
            print(f"{i+1}. [Idx: {idx}] (Score: {score:.4f}) {doc_preview}...")

if __name__ == "__main__":
    debug()

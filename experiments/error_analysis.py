import sys
import os
import json

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import load_documents
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index
from sentence_transformers import SentenceTransformer

from experiments.benchmark_data import benchmark_queries

try:
    from app.reranker import get_reranker
except ImportError:
    get_reranker = None

def run_error_analysis():
    print("Loading ArXiv ML Papers Dataset...")
    docs, labels, label_names = load_documents()
    
    # We load a decent chunk to have meaningful retrieval
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
            # Force load
            _ = reranker.model
            print("\nCross-Encoder Reranker loaded successfully.")
        except Exception as e:
            print(f"\nFailed to load reranker: {e}")
    else:
        print("\nReranker module not available.")

    print(f"\nRunning Error Analysis on {len(benchmark_queries)} natural language queries...")
    
    error_report = []
    
    cat_to_id = {name: i for i, name in enumerate(label_names)}
    for item in benchmark_queries:
        q_text = item["query"]
        relevant_docs = item.get("relevant_docs", [])
        
        # Filter out out-of-bound IDs
        relevant_docs = [doc_id for doc_id in relevant_docs if doc_id < len(docs)]
        
        if not relevant_docs:
            continue
        
        # 1. Retrieve using Hybrid (baseline)
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        hybrid_indices, hybrid_scores, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=10)
        
        # Check top-3 hits for hybrid
        hybrid_top3 = [int(idx) for idx in hybrid_indices[:3]]
        hybrid_hit = any(idx in relevant_docs for idx in hybrid_top3)
        
        # 2. Rerank top 30
        reranked_hit = False
        reranked_top3 = []
        if reranker is not None:
            cand_indices, _, _ = hybrid.search(q_text, q_emb, faiss_index, docs, top_k=30)
            cand_docs = [docs[idx] for idx in cand_indices]
            r_indices, r_scores = reranker.rerank(q_text, cand_docs, cand_indices, top_k=10)
            
            reranked_top3 = [int(idx) for idx in r_indices[:3]]
            reranked_hit = any(idx in relevant_docs for idx in reranked_top3)
            
        # If both hit, no error to analyze. If one hits and other misses, or both miss, we analyze.
        if hybrid_hit and (reranker is None or reranked_hit):
            continue
            
        # Analyze failure
        failure_type = "UNKNOWN"
        
        # What categories DID we retrieve?
        retrieved_cats_hybrid = [label_names[labels[idx]] for idx in hybrid_top3]
        retrieved_cats_reranked = [label_names[labels[idx]] for idx in reranked_top3] if reranked_top3 else []
        
        if not hybrid_hit and reranked_hit:
            failure_type = "HYBRID_MISS_RERANKER_FIXED"
            reason = "Hybrid search failed to push relevant docs to top 3, but reranker corrected it."
        elif hybrid_hit and not reranked_hit:
            failure_type = "RERANKER_DEGRADATION"
            reason = "Reranker pushed irrelevant documents above the relevant ones found by hybrid."
        elif not hybrid_hit and not reranked_hit:
            if set(retrieved_cats_hybrid) == set([expected_cat_name]):
                 failure_type = "DATASET_LABEL_ISSUE"
                 reason = "Retrieved documents look perfectly relevant but ground truth labels don't match."
            else:
                 # Check if the query is ambiguous between two categories
                 if "cs.LG" in expected_cat_name and any("cs." in c for c in retrieved_cats_hybrid):
                     failure_type = "BOUNDARY_CASE"
                     reason = f"Query is ambiguous between {expected_cat_name} and {set(retrieved_cats_hybrid)}."
                 else:
                     failure_type = "RETRIEVAL_FAILURE"
                     reason = "Neither hybrid nor reranker found relevant documents in top 3."
        
        error_report.append({
            "query": q_text,
            "expected_category": expected_cat_name,
            "hybrid_retrieved_categories": retrieved_cats_hybrid,
            "reranked_retrieved_categories": retrieved_cats_reranked,
            "failure_type": failure_type,
            "reason": reason
        })
        
    print(f"\nFound {len(error_report)} queries with retrieval issues out of {len(benchmark_queries)}.")
    
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/error_analysis.json", "w") as f:
        json.dump(error_report, f, indent=2)
        
    print("\n--- Error Analysis Summary ---")
    for err in error_report:
        print(f"\nQuery: {err['query']}")
        print(f"Expected: {err['expected_category']}")
        print(f"Hybrid Top-3 Cats: {err['hybrid_retrieved_categories']}")
        if err['reranked_retrieved_categories']:
            print(f"Reranked Top-3 Cats: {err['reranked_retrieved_categories']}")
        print(f"Failure Type: {err['failure_type']}")
        print(f"Reason: {err['reason']}")
        print("-" * 40)
        
    print("\nError analysis report saved to experiments/results/error_analysis.json")

if __name__ == "__main__":
    run_error_analysis()

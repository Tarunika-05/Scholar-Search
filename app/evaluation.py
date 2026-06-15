import numpy as np
from typing import Union, Dict, List

def _get_relevance(relevant_docs: Union[List[int], Dict[int, float]], doc_id: int) -> float:
    """Helper to get relevance score regardless of input format."""
    if isinstance(relevant_docs, dict):
        return relevant_docs.get(doc_id, 0.0)
    return 1.0 if doc_id in relevant_docs else 0.0

def _is_relevant(relevant_docs: Union[List[int], Dict[int, float]], doc_id: int) -> bool:
    """Helper to check if a document has non-zero relevance."""
    return _get_relevance(relevant_docs, doc_id) > 0.0

def precision_at_k(relevant_docs: Union[List[int], Dict[int, float]], retrieved_docs: List[int], k: int) -> float:
    """
    Computes Precision@K.
    """
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_docs[:k]
    hits = sum(1 for doc_id in retrieved_k if _is_relevant(relevant_docs, doc_id))
    return hits / k


def mrr(relevant_docs: Union[List[int], Dict[int, float]], retrieved_docs: List[int]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for a single query.
    """
    for rank, doc_id in enumerate(retrieved_docs, 1):
        if _is_relevant(relevant_docs, doc_id):
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevant_docs: Union[List[int], Dict[int, float]], retrieved_docs: List[int], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at K.
    Supports graded relevance natively.
    """
    retrieved_k = retrieved_docs[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        rel_score = _get_relevance(relevant_docs, doc_id)
        if rel_score > 0:
            dcg += (2 ** rel_score - 1) / np.log2(i + 2)
    return dcg


def ndcg_at_k(relevant_docs: Union[List[int], Dict[int, float]], retrieved_docs: List[int], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K.
    """
    dcg = dcg_at_k(relevant_docs, retrieved_docs, k)
    
    # Ideal DCG is when all relevant items are sorted by relevance.
    if isinstance(relevant_docs, dict):
        ideal_scores = sorted(relevant_docs.values(), reverse=True)
    else:
        ideal_scores = [1.0] * len(relevant_docs)
        
    idcg = 0.0
    ideal_scores_k = ideal_scores[:k]
    for i, rel_score in enumerate(ideal_scores_k):
        if rel_score > 0:
            idcg += (2 ** rel_score - 1) / np.log2(i + 2)
            
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg


def recall_at_k(relevant_docs: Union[List[int], Dict[int, float]], retrieved_docs: List[int], k: int) -> float:
    """
    Computes Recall@K: fraction of relevant documents found in the top-K results.
    """
    if len(relevant_docs) == 0:
        return 0.0
        
    total_relevant = len(relevant_docs) if not isinstance(relevant_docs, dict) else sum(1 for v in relevant_docs.values() if v > 0)
    if total_relevant == 0:
        return 0.0
        
    retrieved_k = retrieved_docs[:k]
    hits = sum(1 for doc_id in retrieved_k if _is_relevant(relevant_docs, doc_id))
    return hits / total_relevant


def average_precision(relevant_docs: Union[List[int], Dict[int, float]], retrieved_docs: List[int]) -> float:
    """
    Computes Average Precision (AP) for a single query.
    """
    if len(relevant_docs) == 0:
        return 0.0
        
    total_relevant = len(relevant_docs) if not isinstance(relevant_docs, dict) else sum(1 for v in relevant_docs.values() if v > 0)
    if total_relevant == 0:
        return 0.0
        
    hits = 0
    sum_precision = 0.0
    
    for rank, doc_id in enumerate(retrieved_docs, 1):
        if _is_relevant(relevant_docs, doc_id):
            hits += 1
            sum_precision += hits / rank
            
    return sum_precision / total_relevant


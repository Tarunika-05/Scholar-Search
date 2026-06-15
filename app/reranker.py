import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Cross-encoder reranking stage for retrieval.
    
    Why cross-encoders beat bi-encoders for reranking:
    - Bi-encoders (like all-MiniLM-L6-v2) encode query and document independently.
      They can't model fine-grained query-document interactions.
    - Cross-encoders take the (query, document) pair as joint input, allowing
      full cross-attention between query and document tokens.
    - This gives much better relevance scores, but is too slow for first-stage
      retrieval (O(n) comparisons vs O(1) for bi-encoder + ANN index).
    
    Architecture: Retrieve top-100 with fast hybrid search → Rerank top-10 with cross-encoder.
    This gives cross-encoder quality at bi-encoder speed.
    
    Model: BAAI/bge-reranker-v2-m3 (568M params, multilingual, domain-agnostic)
    
    Why bge-reranker-v2-m3 over ms-marco-MiniLM-L-6-v2:
    - ms-marco-MiniLM was trained exclusively on Bing web search queries.
      It fails on domain-specific text (scientific papers, legal docs, etc.)
      because it doesn't recognise specialised terminology.
    - bge-reranker-v2-m3 is trained on diverse multilingual data and generalises
      across domains (web, scientific, technical) without domain shift.
    - On BEIR benchmarks, bge-reranker consistently outperforms ms-marco cross-encoders.
    """
    
    _instance = None  # Singleton for lazy loading
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None  # Lazy load
    
    @property
    def model(self):
        """Lazy load the cross-encoder model on first use."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded.")
        return self._model
    
    def rerank(self, query: str, candidate_docs: list[str], 
              candidate_indices: list[int], top_k: int = 5) -> tuple[list[int], list[float]]:
        """
        Rerank candidate documents using cross-encoder scores natively.
        
        Args:
            query: The search query string
            candidate_docs: List of candidate document texts from first-stage retrieval
            candidate_indices: Original corpus indices of the candidate documents
            top_k: Number of top results to return after reranking
            
        Returns:
            reranked_indices: Document indices sorted by cross-encoder score
            reranked_scores: Cross-encoder scores for each reranked document
        """
        if not candidate_docs:
            return [], []
        
        # Build (query, document) pairs for the cross-encoder
        pairs = [[query, doc] for doc in candidate_docs]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        scores = scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
        
        # Sort purely by the powerful BGE reranker score descending
        scored = sorted(
            zip(candidate_indices, scores, candidate_docs),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top_k
        top = scored[:top_k]
        reranked_indices = [item[0] for item in top]
        reranked_scores = [round(float(item[1]), 4) for item in top]
        
        return reranked_indices, reranked_scores


# Module-level singleton for reuse across requests
_reranker = None

def get_reranker() -> CrossEncoderReranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker

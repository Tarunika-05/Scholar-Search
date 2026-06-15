import faiss
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────────────────────────────
# Vector Database Choice: FAISS (Facebook AI Similarity Search)
#
# Why FAISS:
# 1. Efficiency — FAISS is optimised for nearest-neighbor search on
#    dense vectors. For 5000 documents at 384 dimensions, it handles
#    search in microseconds.
# 2. No server required — unlike Pinecone, Weaviate, or Qdrant, FAISS
#    runs fully in-process. No network overhead, no setup complexity.
# 3. Battle-tested — used in production at Facebook/Meta scale.
# 4. Flexible index types — I use IndexFlatIP (Inner Product) since
#    my embeddings are L2-normalized, making inner product equivalent
#    to cosine similarity. This is the most accurate index type.
#
# Trade-off acknowledged:
# FAISS IndexFlatIP does exhaustive search — it checks every vector.
# For 5000 docs this is fine (<1ms). For millions of docs, I would
# switch to IndexIVFFlat (approximate) for speed at slight accuracy cost.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Metadata-Aware Vector Store (Novelty)
#
# Plain FAISS stores vectors but knows nothing about documents.
# For filtered retrieval (e.g. "search only in politics category"),
# I need metadata attached to each vector.
#
# my approach:
# - Wrap IndexFlatIP with IndexIDMap so vectors get stable integer IDs
# - Maintain a parallel metadata dict mapping doc_id → {category, ...}
# - search_with_filter() does FAISS top-k then post-filters by metadata
#
# Why post-filtering instead of pre-filtering?
# Pre-filtering would require building separate FAISS indices per
# category (e.g. 5 indices for 5 ArXiv categories). This wastes memory and
# complicates cross-category queries. A single unified index is better.
# Post-filtering on a 5000-doc corpus is negligible — I over-fetch by 2x
# and filter, still <1ms. For million-scale corpora, pre-filtering or
# partitioned indices would be the right call.
# ─────────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "data/faiss_index.pkl"


def build_index(embeddings: np.ndarray, labels: list | None = None) -> dict:
    """
    Build a FAISS index with optional metadata for filtered retrieval.

    I use IndexIDMap wrapping IndexFlatIP because:
    - IndexFlatIP gives exact cosine similarity (on L2-normalized vectors)
    - IndexIDMap lets us assign stable document IDs that survive
      index rebuilds and map to my metadata dictionary
    - This enables filtered retrieval without separate per-category indices

    Args:
        embeddings: numpy array of shape (n_docs, 384), L2-normalized
        labels:     optional list of integer category labels per document

    Returns:
        dict with keys:
            'index':    FAISS IndexIDMap wrapping IndexFlatIP
            'metadata': dict mapping doc_id → {category, text_length}
    """
    dimension = embeddings.shape[1]  # 384 for MiniLM
    n_docs = len(embeddings)

    print(f"Building FAISS index (dimension={dimension}, docs={n_docs})...")

    # Build base index + ID mapping
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)

    # FAISS requires float32
    embeddings_f32 = np.array(embeddings, dtype=np.float32)
    # Assign sequential IDs: 0, 1, 2, ..., n_docs-1
    ids = np.arange(n_docs, dtype=np.int64)
    index.add_with_ids(embeddings_f32, ids)

    # Build metadata — stores per-document attributes for filtered retrieval.
    # This is a plain dict because at 5000 docs the overhead is trivial.
    # For larger corpora, this could be backed by SQLite or a DataFrame.
    metadata = {}
    for i in range(n_docs):
        metadata[i] = {
            "category": int(labels[i]) if labels is not None else -1,
        }

    print(f"[SUCCESS] FAISS index built. Total vectors: {index.ntotal}")
    if labels is not None:
        n_categories = len(set(labels))
        print(f"   Metadata: {n_categories} categories indexed for filtered retrieval")

    return {"index": index, "metadata": metadata}


def search_index(index_data, query_embedding: np.ndarray, limit: int = 5, offset: int = 0):
    """
    Search the FAISS index for the most similar documents.

    Handles both old-style (raw FAISS index) and new-style (dict with
    index + metadata) for backward compatibility.

    Args:
        index_data:      FAISS index or dict with 'index' + 'metadata'
        query_embedding: 1D numpy array of shape (384,), L2-normalized
        limit:           Number of results to return
        offset:          Number of results to skip (pagination)

    Returns:
        distances: similarity scores (cosine similarity since normalized)
        indices:   document indices in the original corpus
    """
    # Support both raw index and dict wrapper
    index = index_data["index"] if isinstance(index_data, dict) else index_data

    # We need to fetch offset + limit documents to skip the first 'offset' ones
    fetch_k = offset + limit
    
    # FAISS expects shape (1, dimension) for single query
    query = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query, fetch_k)

    # Flatten and slice for pagination
    return distances[0][offset:], indices[0][offset:]


def search_with_filter(index_data: dict, query_embedding: np.ndarray,
                       category_filter: int | None = None, limit: int = 5, offset: int = 0):
    """
    Search FAISS with optional metadata-based category filtering and pagination.

    Strategy: over-fetch from FAISS, then post-filter by metadata.

    Args:
        index_data:       dict with 'index' + 'metadata'
        query_embedding:  L2-normalized query vector (384-dim)
        category_filter:  integer category ID to filter by, or None for unfiltered
        limit:            number of filtered results to return
        offset:           number of filtered results to skip

    Returns:
        filtered_distances: similarity scores for matching documents
        filtered_indices:   document indices of matching documents
    """
    if category_filter is None:
        # No filter — standard search
        return search_index(index_data, query_embedding, limit, offset)

    metadata = index_data.get("metadata", {})

    # Over-fetch to ensure enough candidates survive filtering.
    # 3x multiplier handles skewed category distributions.
    # We must fetch enough to cover both offset and limit AFTER filtering.
    over_fetch = (offset + limit) * 3
    distances, indices = search_index(index_data, query_embedding, limit=over_fetch, offset=0)

    # Post-filter by category
    filtered_distances = []
    filtered_indices = []

    for dist, idx in zip(distances, indices):
        if idx < 0:
            continue  # FAISS returns -1 for empty slots
        doc_meta = metadata.get(int(idx), {})
        if doc_meta.get("category") == category_filter:
            filtered_distances.append(dist)
            filtered_indices.append(idx)
            
    # Apply pagination AFTER filtering
    final_distances = filtered_distances[offset:offset+limit]
    final_indices = filtered_indices[offset:offset+limit]

    return (
        np.array(final_distances, dtype=np.float32),
        np.array(final_indices, dtype=np.int64)
    )


def save_index(index_data):
    """
    Persist FAISS index and metadata to disk.
    Rebuilding the index on every startup would add ~5s delay.
    Metadata is stored alongside so filtered retrieval works after reload.
    """
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    print(f"[SUCCESS] FAISS index + metadata saved to {FAISS_INDEX_PATH}")


def load_index():
    """
    Load FAISS index (and metadata if present) from disk.
    Returns None if not found.

    Handles both legacy format (raw FAISS index) and new format
    (dict with 'index' + 'metadata') for backward compatibility.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return None

    print("Loading cached FAISS index from disk...")
    with open(FAISS_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    # Handle legacy format (raw index without metadata)
    if isinstance(data, dict):
        print(f"[SUCCESS] FAISS index loaded. Total vectors: {data['index'].ntotal}")
        n_categories = len(set(m.get('category', -1) for m in data.get('metadata', {}).values()))
        print(f"   Metadata: {n_categories} categories available for filtered retrieval")
        return data
    else:
        # Legacy: raw FAISS index, wrap it
        print(f"[SUCCESS] FAISS index loaded (legacy format). Total vectors: {data.ntotal}")
        return {"index": data, "metadata": {}}

import pytest
import numpy as np
from app.vector_store import build_index, search_index, search_with_filter

@pytest.fixture
def sample_index_data():
    np.random.seed(42)
    # 20 docs, 384 dim
    embeddings = np.random.randn(20, 384).astype(np.float32)
    labels = [i % 4 for i in range(20)] # 4 categories
    return build_index(embeddings, labels), embeddings

def test_search_index_pagination(sample_index_data):
    index_data, embeddings = sample_index_data
    query = embeddings[0] # Exact match for doc 0
    
    # Fetch first 5
    dists1, idxs1 = search_index(index_data, query, limit=5, offset=0)
    assert len(idxs1) == 5
    assert idxs1[0] == 0
    
    # Fetch next 5
    dists2, idxs2 = search_index(index_data, query, limit=5, offset=5)
    assert len(idxs2) == 5
    # Ensure disjoint
    assert len(set(idxs1).intersection(set(idxs2))) == 0

def test_search_with_filter_pagination(sample_index_data):
    index_data, embeddings = sample_index_data
    query = embeddings[0]
    
    # Filter by category 0 (docs 0, 4, 8, 12, 16)
    dists, idxs = search_with_filter(index_data, query, category_filter=0, limit=2, offset=0)
    assert len(idxs) == 2
    for idx in idxs:
        assert idx % 4 == 0
        
    dists2, idxs2 = search_with_filter(index_data, query, category_filter=0, limit=2, offset=2)
    assert len(idxs2) <= 2 # Depending on how many are left, could be 2 or less
    # Ensure disjoint
    assert len(set(idxs).intersection(set(idxs2))) == 0

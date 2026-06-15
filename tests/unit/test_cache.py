import pytest
import numpy as np
import time
from app.cache import SemanticCache

@pytest.fixture
def cache():
    return SemanticCache(threshold=0.85, max_entries=5)

def test_empty_cache_returns_none(cache):
    query_emb = np.array([1.0, 0.0], dtype=np.float32)
    assert cache.lookup(query_emb, candidate_indices=[]) is None

def test_store_and_lookup_exact_match(cache):
    query_emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.store("test query", query_emb, "test answer", 0, generated_answer="test answer", citations=[{"id": 1}])
    
    # Simulate lookup with candidate index 0
    result = cache.lookup(query_emb, [0])
    assert result is not None
    assert result.query == "test query"
    assert result.generated_answer == "test answer"

def test_lookup_returns_none_below_threshold(cache):
    emb1 = np.array([1.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0], dtype=np.float32) # Orthogonal, dot product = 0.0
    
    cache.store("query 1", emb1, "ans 1", 0)
    
    # lookup with emb2 should fail threshold (0.0 < 0.85)
    assert cache.lookup(emb2, [0]) is None

def test_lru_eviction_removes_oldest():
    cache = SemanticCache(threshold=0.85, max_entries=2)
    emb1 = np.array([1.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0], dtype=np.float32)
    emb3 = np.array([0.707, 0.707], dtype=np.float32)
    
    cache.store("q1", emb1, "a1", 0)
    cache.store("q2", emb2, "a2", 1)
    cache.store("q3", emb3, "a3", 2) # This should evict q1 (index 0)
    
    # q1 should be evicted
    assert 0 not in cache._store
    assert len(cache._store) == 2
    assert 1 in cache._store
    assert 2 in cache._store

def test_ttl_expiry_skips_stale_entries():
    cache = SemanticCache(threshold=0.85, max_entries=5, ttl_seconds=0.1)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    
    cache.store("q1", emb, "a1", 0)
    time.sleep(0.2) # Wait for expiry
    
    # lookup should return None and delete the stale entry
    assert cache.lookup(emb, [0]) is None
    assert 0 not in cache._store

def test_flush_clears_all_entries_and_stats(cache):
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.store("q1", emb, "a1", 0)
    
    # Force some stats
    cache.lookup(emb, [0]) # Hit
    cache.lookup(np.array([0.0, 1.0], dtype=np.float32), [0]) # Miss
    
    cache.flush()
    
    assert len(cache._store) == 0
    assert len(cache._cluster_index) == 0
    
    stats = cache.get_stats()
    assert stats["entries"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0

def test_set_threshold_validates_range(cache):
    cache.set_threshold(0.9)
    assert cache.threshold == 0.9
    
    with pytest.raises(ValueError):
        cache.set_threshold(1.5)
        
    with pytest.raises(ValueError):
        cache.set_threshold(-0.1)

def test_get_stats_returns_correct_counts(cache):
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.store("q1", emb, "a1", 0)
    
    cache.lookup(emb, [0]) # Hit
    cache.lookup(np.array([0.0, 1.0], dtype=np.float32), [0]) # Miss
    
    stats = cache.get_stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["threshold"] == 0.85

def test_cluster_index_updated_on_store(cache):
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.store("q1", emb, "a1", 5, np.array([0.1, 0.9]))
    
    assert 5 in cache._cluster_index
    assert 0 in cache._cluster_index[5] # Entry ID 0

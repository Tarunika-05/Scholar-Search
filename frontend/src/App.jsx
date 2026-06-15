import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import ResultsPanel from './components/ResultsPanel';
import AdminPanel from './components/AdminPanel';
import CacheStats from './components/CacheStats';
import { searchApi, adminApi } from './api/client';
import { Database } from 'lucide-react';

function App() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const [activeTab, setActiveTab] = useState('search');
  const [stats, setStats] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [threshold, setThreshold] = useState(0.85);
  const [updateMessage, setUpdateMessage] = useState('');
  
  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState('dense');
  const [searchCategory, setSearchCategory] = useState(2);

  const fetchData = async () => {
    try {
      const [cacheRes, analyticsRes] = await Promise.all([
        adminApi.getCacheStats(),
        adminApi.getAnalytics()
      ]);
      setStats(cacheRes.data);
      setAnalytics(analyticsRes.data);
    } catch (err) {
      console.error("Failed to fetch admin data", err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUpdateThreshold = async (newThreshold) => {
    try {
      await adminApi.updateThreshold(newThreshold);
      setThreshold(newThreshold);
      setUpdateMessage(`Threshold updated to ${newThreshold}`);
      setTimeout(() => setUpdateMessage(''), 3000);
      fetchData();
    } catch (err) {
      console.error("Failed to update threshold", err);
      setUpdateMessage('Failed to update threshold');
      setTimeout(() => setUpdateMessage(''), 3000);
    }
  };

  const handleSearch = async (query, mode, category) => {
    setIsLoading(true);
    setHasSearched(true);
    try {
      let res;
      const payload = { query, generate: true };
      
      if (mode === 'dense') {
        res = await searchApi.query(payload);
      } else if (mode === 'hybrid') {
        res = await searchApi.hybridQuery(payload);
      } else if (mode === 'filtered') {
        res = await searchApi.filteredQuery({ ...payload, category });
      }
      
      setResults(res.data);
      fetchData(); // Refresh stats immediately after search
    } catch (error) {
      console.error("Search failed:", error);
      setResults(null);
    }
    setIsLoading(false);
  };

  return (
    <div className="app-container">
      <div className="background-glow"></div>
      
      <header className="app-header">
        <div className="logo-container">
          <Database size={28} className="logo-icon" />
          <h1>Cognitive RAG <span>Explorer</span></h1>
        </div>
        <div className="nav-tabs">
          <button 
            className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            Search
          </button>
          <button 
            className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            Analytics Dashboard
          </button>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'search' ? (
          <div className="single-column fade-in">
            {searchMode === 'dense' && (
              <div className="search-header-stats fade-in" style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem' }}>
                <div className="glass-panel" style={{ padding: '0.5rem 1rem', display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <label style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', whiteSpace: 'nowrap' }}>Semantic Threshold: <b>{threshold}</b></label>
                    <div style={{ display: 'flex', gap: '0.5rem', flex: 1, alignItems: 'center' }}>
                      <input 
                        type="range" 
                        min="0.1" 
                        max="1.0" 
                        step="0.01" 
                        value={threshold} 
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        style={{ flex: 1, height: '4px' }}
                      />
                      <button onClick={() => handleUpdateThreshold(threshold)} className="search-button" style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem' }}>Set</button>
                    </div>
                  </div>
                  {updateMessage && <span style={{ color: 'var(--accent-primary)', fontSize: '0.7rem', marginTop: '0.25rem' }}>{updateMessage}</span>}
                </div>
              </div>
            )}

            <SearchBar 
              onSearch={handleSearch} 
              isLoading={isLoading} 
              query={searchQuery}
              setQuery={setSearchQuery}
              mode={searchMode}
              setMode={setSearchMode}
              category={searchCategory}
              setCategory={setSearchCategory}
            />
            <ResultsPanel results={results} isLoading={isLoading} hasSearched={hasSearched} />
          </div>
        ) : (
          <div className="single-column fade-in" style={{ height: 'calc(100vh - 120px)' }}>
            <AdminPanel stats={stats} analytics={analytics} fetchData={fetchData} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

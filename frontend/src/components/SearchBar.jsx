import React from 'react';
import { Search } from 'lucide-react';

const SearchBar = ({ 
  onSearch, 
  isLoading, 
  query, 
  setQuery, 
  mode, 
  setMode, 
  category, 
  setCategory 
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query, mode, category);
    }
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="search-input-wrapper">
          <Search className="search-icon" size={20} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about machine learning..."
            className="search-input"
            disabled={isLoading}
          />
        </div>
        
        <div className="search-controls">
          <select 
            value={mode} 
            onChange={(e) => setMode(e.target.value)}
            className="mode-select"
            disabled={isLoading}
          >
            <option value="dense">Dense Search</option>
            <option value="hybrid">Hybrid Search</option>
            <option value="filtered">Filtered Search</option>
          </select>
          
          {mode === 'filtered' && (
            <select 
              value={category} 
              onChange={(e) => setCategory(parseInt(e.target.value))}
              className="category-select"
              disabled={isLoading}
            >
              <option value={2}>Machine Learning (cs.LG)</option>
              <option value={1}>Computer Vision (cs.CV)</option>
              <option value={0}>Computation and Language (cs.CL)</option>
            </select>
          )}
          
          <button type="submit" className="search-button" disabled={isLoading || !query.trim()}>
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      <div className="sample-queries">
        <span className="sample-label">Try asking:</span>
        <div className="chips-container">
          {[
            "What is a Large Language Model?",
            "Explain the concept of Large Language Models",
            "How do Convolutional Neural Networks apply to image classification?"
          ].map((sample, idx) => (
            <button 
              key={idx} 
              className="chip" 
              disabled={isLoading}
              onClick={() => {
                setQuery(sample);
                // Intentionally NOT auto-clicking search so user can review the query
              }}
            >
              {sample}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SearchBar;

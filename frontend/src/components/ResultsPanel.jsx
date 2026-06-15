import React from 'react';
import { Bot, FileText, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ResultsPanel = ({ results, isLoading, hasSearched }) => {
  if (isLoading) {
    return (
      <div className="results-loading">
        <div className="pulse-circle"></div>
        <p>Retrieving and synthesizing knowledge...</p>
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="results-empty">
        <Bot size={48} className="empty-icon" />
        <h2>Cognitive RAG Search</h2>
        <p>Ask a complex ML question to search the ArXiv database.</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-empty error">
        <h2>No Results</h2>
        <p>Something went wrong or no documents matched your query.</p>
      </div>
    );
  }

  return (
    <div className="results-panel fade-in">
      {/* Generated Answer Section */}
      {results.generated_answer && (
        <div className="answer-card glass-panel">
          <div className="card-header">
            <Bot size={20} className="text-primary" />
            <h3>AI Synthesis</h3>
            {results.cache_hit ? (
              <span className="badge cache-badge">
                <Zap size={12} /> Semantic Cache Hit
              </span>
            ) : (
              <span className="badge miss-badge">
                Semantic Cache Miss
              </span>
            )}
          </div>
          <div className="search-metadata">
            <span className="meta-tag">Mode: {results.search_mode}</span>
            <span className="meta-tag">Cluster: {results.dominant_cluster}</span>
          </div>
          <div className="answer-content markdown-body">
            <ReactMarkdown>{results.generated_answer}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* Citations/Retrieved Documents Section */}
      <div className="documents-section">
        <div className="section-header">
          <FileText size={18} />
          <h4>Retrieved Context Sources</h4>
        </div>
        
        {results.citations ? (
          <div className="citations-list">
            {results.citations.map((cite, idx) => (
              <div key={idx} className="citation-card glass-panel">
                <div className="citation-title">[{idx + 1}] {cite.title}</div>
                {cite.snippet && <div className="citation-snippet">{cite.snippet}</div>}
                <div className="citation-meta">
                  <span className="doc-id">Doc ID: {cite.paper_id}</span>
                  <span className="score">Score: {cite.retrieval_score.toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="raw-results glass-panel">
            <pre>{results.result}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsPanel;

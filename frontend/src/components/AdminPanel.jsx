import React, { useState, useEffect, useRef } from 'react';
import { adminApi } from '../api/client';
import { Activity, Trash2, RefreshCw } from 'lucide-react';
import CacheStats from './CacheStats';
import LatencyChart from './LatencyChart';
import ClusterViz from './ClusterViz';

const AdminPanel = ({ stats, analytics, fetchData }) => {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalResults, setEvalResults] = useState(null);
  const [adminMessage, setAdminMessage] = useState('');

  const evalRef = useRef(null);

  const handleClearCache = async () => {
    try {
      await adminApi.clearCache();
      setAdminMessage('Cache cleared successfully!');
      setTimeout(() => setAdminMessage(''), 3000);
      fetchData();
    } catch (err) {
      console.error("Failed to clear cache", err);
    }
  };

  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const res = await adminApi.runEvaluation();
      setEvalResults(res.data);
      setTimeout(() => {
        if (evalRef.current) {
          evalRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } catch (err) {
      console.error("Eval failed", err);
    }
    setIsEvaluating(false);
  };

  return (
    <div className="admin-panel glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="admin-header">
        <h3><Activity size={18}/> System Dashboard</h3>
        <button className="icon-button" onClick={fetchData}><RefreshCw size={14}/></button>
      </div>

      <div className="charts-grid" style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem' }}>
        <CacheStats stats={stats} analytics={analytics} />
        <LatencyChart analytics={analytics} />
        <div style={{ gridColumn: '1 / -1' }}>
          <ClusterViz />
        </div>
        
        {evalResults && !evalResults.error && (
          <div style={{ gridColumn: '1 / -1' }} ref={evalRef}>
            <div className="eval-results glass-panel fade-in" style={{ marginTop: '1rem', overflowX: 'auto' }}>
              <h4>Retrieval Evaluation (NDCG/MRR)</h4>
              <table className="eval-table">
                <thead>
                  <tr>
                    <th>Mode</th>
                    <th>NDCG@10</th>
                    <th>MRR</th>
                    <th>P@3</th>
                    <th>R@10</th>
                    <th>MAP</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(evalResults).map(([mode, metrics]) => (
                    <tr key={mode}>
                      <td>{mode}</td>
                      <td>{metrics['ndcg@10'].toFixed(4)}</td>
                      <td>{metrics['mrr'].toFixed(4)}</td>
                      <td>{metrics['p@3'].toFixed(4)}</td>
                      <td>{metrics['r@10'].toFixed(4)}</td>
                      <td>{metrics['map'].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {evalResults?.error && (
          <div style={{ gridColumn: '1 / -1' }}>
            <div className="eval-error">{evalResults.error}</div>
          </div>
        )}
      </div>

      <div className="admin-controls" style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
        
        <div className="button-group" style={{marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '1rem'}}>
          <button onClick={handleClearCache} className="btn-danger">
            <Trash2 size={14}/> Clear Cache
          </button>
          <button onClick={handleRunEvaluation} className="btn-primary" disabled={isEvaluating}>
            {isEvaluating ? 'Evaluating...' : 'Load IR Metrics'}
          </button>
          {adminMessage && <span style={{ color: 'var(--accent-primary)', fontSize: '0.8rem' }}>{adminMessage}</span>}
        </div>
      </div>
    </div>
  );
};

export default AdminPanel;

import React, { useEffect, useState } from 'react';
import { adminApi } from '../api/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

const ClusterViz = () => {
  const [clusterData, setClusterData] = useState(null);

  useEffect(() => {
    const fetchClusters = async () => {
      try {
        const res = await adminApi.getClustersAnalysis();
        setClusterData(res.data);
      } catch (err) {
        console.error("Failed to load clusters", err);
      }
    };
    
    fetchClusters();
    const interval = setInterval(() => {
      // Only poll if we don't have data yet, to recover from server restarts
      if (!clusterData) {
        fetchClusters();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [clusterData]);

  if (!clusterData || !clusterData.clusters) {
    return (
      <div className="chart-container glass-panel fade-in" style={{ marginTop: '1rem', padding: '1rem', height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-secondary)' }}>Loading cluster data...</p>
      </div>
    );
  }

  // Format data for Recharts
  const data = Object.entries(clusterData.clusters)
    .map(([id, info]) => ({
      name: `C${id}: ${info.topic_words[0] || 'Unknown'}`,
      size: info.size,
      topics: info.topic_words.slice(0, 3).join(', ')
    }))
    .sort((a, b) => b.size - a.size) // sort by size descending
    .slice(0, 10); // show top 10 clusters for cleaner UI

  return (
    <div className="chart-container glass-panel fade-in" style={{ marginTop: '1rem', padding: '1rem' }}>
      <h4><span style={{color: '#9ca3af', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px'}}>Topic Modeling</span><br/>Largest GMM Clusters</h4>
      <div style={{ height: 250, marginTop: '1rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
            <XAxis dataKey="name" stroke="#9ca3af" fontSize={11} interval={0} angle={-30} textAnchor="end" height={60} />
            <YAxis stroke="#9ca3af" fontSize={11} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: '#374151', borderRadius: '8px' }}
              itemStyle={{ color: '#e5e7eb' }}
              cursor={{fill: 'rgba(255, 255, 255, 0.05)'}}
              formatter={(value, name, props) => [value, props.payload.topics]}
            />
            <Bar dataKey="size" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ClusterViz;

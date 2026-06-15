import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

const CacheStats = ({ stats, analytics, minimal = false }) => {
  if (!stats || !analytics) return null;

  const hits = analytics.cache_hits || 0;
  const misses = (analytics.total_queries || 0) - hits;

  const data = [
    { name: 'Cache Hits', value: hits },
    { name: 'Cache Misses', value: misses },
  ];

  const COLORS = ['#10b981', '#ef4444']; // Green for hit, Red for miss

  return (
    <div className={minimal ? "fade-in" : "chart-container glass-panel fade-in"} style={{ marginTop: minimal ? 0 : '1rem', padding: minimal ? 0 : '1rem', display: 'flex', alignItems: 'center', height: '100%' }}>
      <div style={{ flex: 1 }}>
        <h4 style={{ margin: 0, fontSize: minimal ? '0.8rem' : '1rem' }}><span style={{color: '#9ca3af', fontSize: minimal ? '0.65rem' : '0.8rem', textTransform: 'uppercase', letterSpacing: '1px'}}>Caching</span>{minimal ? ' Stats' : <><br/>Hit/Miss Ratio</>}</h4>
        <div style={{ marginTop: minimal ? '0.2rem' : '0.5rem', display: minimal ? 'flex' : 'block', gap: '1rem' }}>
          <p style={{ margin: 0, fontSize: minimal ? '0.75rem' : '0.9rem', color: '#e5e7eb' }}>Queries: {analytics.total_queries || 0}</p>
          <p style={{ margin: 0, fontSize: minimal ? '0.75rem' : '0.9rem', color: '#e5e7eb' }}>Hit Rate: {analytics.cache_hit_rate || 0}%</p>
          {!minimal && <p style={{ margin: 0, fontSize: '0.9rem', color: '#e5e7eb' }}>Items: {stats.total_entries || 0}</p>}
        </div>
      </div>
      
      <div style={{ width: minimal ? 50 : 120, height: minimal ? 50 : 120, marginLeft: '1rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={30}
              outerRadius={50}
              paddingAngle={5}
              dataKey="value"
              stroke="none"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: '#374151', borderRadius: '8px' }}
              itemStyle={{ color: '#e5e7eb' }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CacheStats;

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

const LatencyChart = ({ analytics }) => {
  if (!analytics) return null;

  const data = [
    { name: 'Cache Hit', ms: analytics.average_latency_hit_ms || 0 },
    { name: 'Cache Miss', ms: analytics.average_latency_miss_ms || 0 },
  ];

  const colors = ['#10b981', '#3b82f6'];

  return (
    <div className="chart-container glass-panel fade-in" style={{ marginTop: '1rem', padding: '1rem' }}>
      <h4><span style={{color: '#9ca3af', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px'}}>Performance</span><br/>Latency Comparison</h4>
      <div style={{ height: 180, marginTop: '1rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={true} vertical={false} />
            <XAxis type="number" stroke="#9ca3af" unit=" ms" />
            <YAxis dataKey="name" type="category" stroke="#9ca3af" width={80} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: '#374151', borderRadius: '8px' }}
              itemStyle={{ color: '#e5e7eb' }}
              cursor={{fill: 'rgba(255, 255, 255, 0.05)'}}
            />
            <Bar dataKey="ms" radius={[0, 4, 4, 0]} barSize={24}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default LatencyChart;

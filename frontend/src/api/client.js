import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'admin-secret-key' // Default admin key
  }
});

export const searchApi = {
  query: (data) => api.post('/query', data),
  hybridQuery: (data) => api.post('/hybrid-query', data),
  filteredQuery: (data) => api.post('/filtered-query', data),
};

export const adminApi = {
  getCacheStats: () => api.get('/cache/stats'),
  clearCache: () => api.delete('/cache'),
  updateThreshold: (threshold) => api.patch(`/cache/threshold?threshold=${threshold}`),
  getAnalytics: () => api.get('/analytics'),
  getClustersAnalysis: () => api.get('/clusters/analysis'),
  runEvaluation: () => api.get('/evaluate'),
};

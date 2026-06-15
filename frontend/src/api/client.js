import axios from 'axios';

// Use environment variable if provided (for production), otherwise default to localhost
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
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

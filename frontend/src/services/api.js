// Axios instance pre-configured for the ARC-FL API
// Handles JWT attachment and 401/error processing

import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor: attach JWT token from localStorage
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('arc_fl_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor: handle 401 (session expired) globally
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      // Clear session and redirect to login
      localStorage.removeItem('arc_fl_token')
      localStorage.removeItem('arc_fl_user')
      window.location.href = '/login'
    }
    
    const message = error.response?.data?.message || error.message || 'An error occurred'
    console.error(`[API Error] ${error.response?.status} ${message}`)
    return Promise.reject(new Error(message))
  }
)

// ============================================================
// Auth endpoints
// ============================================================
export const authApi = {
  login: (email, password) => api.post('/auth/login', { email, password }),
  register: (data) => api.post('/auth/register', data),
  getMe: () => api.get('/auth/me'),
}

// ============================================================
// Client endpoints
// ============================================================
export const clientApi = {
  getDashboard: () => api.get('/client/dashboard'),
  getModel: () => api.get('/client/model'),
  downloadModel: () => `${API_BASE}/client/model/download`,
  getDatasets: () => api.get('/client/datasets'),
  getConfig: () => api.get('/client/config'),
  updateConfig: (data) => api.put('/client/config', data),
  getTrainingMonitor: (params) => api.get('/client/training-monitor', { params: typeof params === 'object' ? params : { roundId: params } }),
  getTrainingStatus: () => api.get('/client/training-status'),
  trainLocal: (data) => api.post('/client/train', data, { timeout: 600_000 }),
  submitTrained: (data) => api.post('/client/submit-trained', data),
  submitUpdate: (formData) => api.post('/client/submit', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300_000,
  }),
  getSubmissions: () => api.get('/client/submissions'),
  getArcHistory: () => api.get('/client/arc-history'),
}

// ============================================================
// Admin endpoints
// ============================================================
export const adminApi = {
  getDashboard: () => api.get('/admin/dashboard'),
  getClients: () => api.get('/admin/clients'),
  updateClientStatus: (clientId, status) =>
    api.patch(`/admin/clients/${clientId}/status`, { status }),
  getTrainingMonitor: (params) => api.get('/admin/training-monitor', { params: typeof params === 'object' ? params : { roundId: params } }),
  getRounds: () => api.get('/admin/rounds'),
  getRoundDetails: (roundId) => api.get(`/admin/rounds/${roundId}`),
  createRound: (data) => api.post('/admin/rounds', data || {}),
  updateRound: (roundId, data) => api.put(`/admin/rounds/${roundId}`, data),
  startRound: (roundId) => api.post(`/admin/rounds/${roundId}/start`),
  cancelRound: (roundId, reason) => api.post(`/admin/rounds/${roundId}/cancel`, { reason }),
  deleteRound: (roundId) => api.delete(`/admin/rounds/${roundId}`),
  getRoundDependencies: (roundId) => api.get(`/admin/rounds/${roundId}/dependencies`),
  archiveRound: (roundId, reason) => api.post(`/admin/rounds/${roundId}/archive`, { reason }),
  getAllSubmissions: (params) => api.get('/admin/submissions', { params }),
  getRoundSubmissions: (roundId) => api.get(`/admin/rounds/${roundId}/submissions`),
  reviewSubmission: (submissionId, status, message) =>
    api.patch(`/admin/submissions/${submissionId}/review`, { status, validationMessage: message }),
  getAnalytics: () => api.get('/admin/analytics'),
  getAuditLogs: () => api.get('/admin/audit-logs'),
  getSystemHealth: () => api.get('/admin/system-health'),
}

// ============================================================
// ARC endpoints
// ============================================================
export const arcApi = {
  getScores: (roundId) => api.get('/arc/scores', { params: { roundId } }),
  getConfig: () => api.get('/arc/config'),
  getConfigHistory: () => api.get('/arc/configs'),
  updateConfig: (data) => api.put('/arc/config', data),
  recalculateScores: (data) => api.post('/arc/recalculate', data || {}),
}

// ============================================================
// Model endpoints
// ============================================================
export const modelApi = {
  getCurrent: () => api.get('/models/current'),
  getReferences: () => api.get('/models/references'),
  getVersions: () => api.get('/models/versions'),
  getHistory: () => api.get('/models/versions'), // Backward compatibility alias
  getDetails: (modelId) => api.get(`/models/details/${modelId}`),
  compare: (modelA, modelB) => api.get('/models/compare', { params: { modelA, modelB } }),
  publish: (modelId) => api.post(`/models/${modelId}/publish`),
  publishCandidate: (data) => api.post('/models/publish-candidate', data),
  publishCandidateAndStartRound: (data) => api.post('/models/publish-candidate-and-start-round', data),
  rejectCandidate: (data) => api.post('/models/reject-candidate', data),
  rollback: (modelId, reason) => api.post(`/models/${modelId}/rollback`, { reason }),
}

// ============================================================
// Aggregation endpoints
// ============================================================
export const aggregationApi = {
  previewWeights: (data) => api.post('/aggregation/preview', data),
  mergeModels: (data) => api.post('/aggregation/merge', data),
  getRoundAggregation: (roundId) => api.get(`/aggregation/round/${roundId}`),
  getHistory: () => api.get('/aggregation/history'),
}

export default api

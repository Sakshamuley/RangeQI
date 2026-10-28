import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// ── Recommendation pipeline (primary) ───────────────────────────────────────
export const recommend   = (payload) => api.post('/recommend', payload).then(r => r.data)

// ── Single-model predict ─────────────────────────────────────────────────────
export const predict     = (payload) => api.post('/predict', payload).then(r => r.data)

// ── Geocoding ────────────────────────────────────────────────────────────────
export const geocode     = (place)   => api.get('/geocode', { params: { place } }).then(r => r.data)

// ── Catalogue ────────────────────────────────────────────────────────────────
export const getEVModels = (params)  => api.get('/ev-models', { params }).then(r => r.data)
export const getBodyTypes = ()       => api.get('/body-types').then(r => r.data)

// ── Data / results ───────────────────────────────────────────────────────────
export const getBenchmark  = ()      => api.get('/benchmark').then(r => r.data)
export const getValidation = ()      => api.get('/validation').then(r => r.data)
export const getHistory    = (limit = 20) => api.get('/history', { params: { limit } }).then(r => r.data)
export const getScan       = (id)    => api.get(`/scan/${id}`).then(r => r.data)

export default api

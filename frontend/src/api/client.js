import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getEVModels = () => api.get('/ev-models').then(r => r.data)
export const predict = (payload) => api.post('/predict', payload).then(r => r.data)
export const getBenchmark = () => api.get('/benchmark').then(r => r.data)
export const getValidation = () => api.get('/validation').then(r => r.data)
export const getScan = (id) => api.get(`/scan/${id}`).then(r => r.data)
export const getHistory = (limit = 20) => api.get('/history', { params: { limit } }).then(r => r.data)
export const getRouteElevation = (params) =>
  api.get('/route-elevation', { params }).then(r => r.data)

export default api

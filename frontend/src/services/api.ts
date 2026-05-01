import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000,
})

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / (progressEvent.total || 1)
      )
      console.log(`Upload progress: ${percentCompleted}%`)
    },
  })
  
  return response.data
}

export async function getAnalysisResult(jobId: string) {
  const response = await api.get(`/result/${jobId}`)
  return response.data
}

export async function checkHealth() {
  const response = await api.get('/health')
  return response.data
}

export async function getModelInfo() {
  const response = await api.get('/model/info')
  return response.data
}

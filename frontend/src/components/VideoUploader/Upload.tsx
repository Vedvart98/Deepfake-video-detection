// import { Upload } from 'lucide-react'
import { useCallback, useState } from 'react'
import { useAnalysis } from '@/hooks/useAnalysis'
import { useNavigate } from 'react-router-dom'

export default function Upload() {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const navigate = useNavigate()
  
  const { upload, status, isLoading } = useAnalysis()
  
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      
      const response = await upload(file)
      navigate(`/results/${response.job_id}`)
    }
  }, [upload, navigate])
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }
  
  const handleDragLeave = () => {
    setIsDragging(false)
  }
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = Array.from(e.dataTransfer.files)
    const videoFile = files.find(file => file.type.startsWith('video/'))
    if (videoFile) {
      onDrop([videoFile])
    }
  }
  
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onDrop(Array.from(files))
    }
  }
  
  return (
    <div className="max-w-4xl mx-auto py-12 px-4">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Analyze Video for Deepfakes
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload a video to detect if it contains manipulated or synthetic content
        </p>
      </div>
      
      <div
        className={`
          dropzone ${isDragging ? 'active' : ''}
          ${isLoading ? 'opacity-50 cursor-wait' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
          id="video-upload"
          disabled={isLoading}
        />
        
        <label htmlFor="video-upload" className="cursor-pointer">
          {previewUrl ? (
            <video
              src={previewUrl}
              className="max-h-64 mx-auto rounded-lg"
              controls
            />
          ) : (
            <div className="py-8">
              {/* <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" /> */}
              <p className="text-lg text-gray-600 dark:text-gray-400">
                Drag and drop a video here, or click to select
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Supported formats: MP4, AVI, MOV, WebM (max 500MB)
              </p>
            </div>
          )}
        </label>
      </div>
      
      {selectedFile && (
        <div className="mt-4 text-center text-sm text-gray-500">
          Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
        </div>
      )}
      
      {isLoading && (
        <div className="mt-6">
          <div className="flex items-center justify-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
            <span className="text-gray-600 dark:text-gray-400">
              {status === 'pending' ? 'Queuing analysis...' : 'Analyzing video...'}
            </span>
          </div>
        </div>
      )}
      
      <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
        <FeatureCard
          icon="🎯"
          title="High Accuracy"
          description="Advanced CNN + ViT hybrid model for precise detection"
        />
        <FeatureCard
          icon="🔍"
          title="Explainable"
          description="Grad-CAM visualizations show exactly what the model sees"
        />
        <FeatureCard
          icon="🔊"
          title="Multi-modal"
          description="Audio-visual consistency checking for enhanced accuracy"
        />
      </div>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="card text-center">
      <div className="text-4xl mb-3">{icon}</div>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        {title}
      </h3>
      <p className="text-gray-600 dark:text-gray-400 text-sm">
        {description}
      </p>
    </div>
  )
}

import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getAnalysisResult } from '@/services/api'
import { CheckCircle, XCircle, AlertCircle, ArrowLeft, Loader, HelpCircle } from 'lucide-react'
import type { AnalysisResponse } from '@/types'

export default function Results() {
  const { jobId } = useParams<{ jobId: string }>()
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['result', jobId],
    queryFn: () => jobId ? getAnalysisResult(jobId) : null,
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as AnalysisResponse | undefined
      if (data?.status === 'pending' || data?.status === 'processing') {
        return 1000
      }
      return false
    },
  })
  
  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto py-12 px-4">
        <div className="card text-center py-12">
          <Loader className="w-12 h-12 mx-auto text-primary-600 animate-spin mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Analyzing Video
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Please wait while we process your video...
          </p>
        </div>
      </div>
    )
  }
  
  if (error || data?.status === 'failed') {
    return (
      <div className="max-w-4xl mx-auto py-12 px-4">
        <div className="card text-center py-12">
          <XCircle className="w-12 h-12 mx-auto text-danger-600 mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Analysis Failed
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            {data?.error_message || 'An error occurred while processing your video.'}
          </p>
          <Link to="/upload" className="btn-primary">
            Try Again
          </Link>
        </div>
      </div>
    )
  }
  
  if (data?.status !== 'completed') {
    return (
      <div className="max-w-4xl mx-auto py-12 px-4">
        <div className="card text-center py-12">
          <AlertCircle className="w-12 h-12 mx-auto text-yellow-600 mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Processing...
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Status: {data?.status}
          </p>
        </div>
      </div>
    )
  }
  
  const prediction = data?.video_prediction
  const isFake = prediction?.label === 'FAKE'
  const isUncertain = prediction?.label === 'UNCERTAIN'
  
  const getVerdictIcon = () => {
    if (isUncertain) return <HelpCircle className="w-20 h-20 mx-auto text-yellow-500 mb-4" />
    return isFake ? (
      <XCircle className="w-20 h-20 mx-auto text-danger-600 mb-4" />
    ) : (
      <CheckCircle className="w-20 h-20 mx-auto text-success-600 mb-4" />
    )
  }
  
  const getVerdictColor = () => {
    if (isUncertain) return 'text-yellow-600'
    return isFake ? 'text-danger-600' : 'text-success-600'
  }
  
  const getBorderColor = () => {
    if (isUncertain) return 'border-yellow-500'
    return isFake ? 'border-danger-500' : 'border-success-500'
  }
  
  const getCardBorderColor = () => {
    if (isUncertain) return 'border-2 border-yellow-500'
    return isFake ? 'border-2 border-danger-500' : 'border-2 border-success-500'
  }
  
  const getVerdictText = () => {
    if (isUncertain) return 'UNCERTAIN - MANUAL REVIEW RECOMMENDED'
    return isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC VIDEO'
  }
  
  const getConfidenceLevelBadge = () => {
    const level = prediction?.confidence_level
    const badges = {
      'HIGH CONFIDENCE': 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-400',
      'LOW CONFIDENCE': 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
      'REVIEW NEEDED': 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
    }
    return badges[level as keyof typeof badges] || 'bg-gray-100 text-gray-700'
  }
  
  return (
    <div className="max-w-4xl mx-auto py-12 px-4">
      <Link
        to="/upload"
        className="inline-flex items-center gap-2 text-gray-600 hover:text-primary-600 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Analyze another video
      </Link>
      
      <div className={`card mb-6 ${getCardBorderColor()}`}>
        <div className="text-center py-8">
          {getVerdictIcon()}
          
          <h1 className={`text-4xl font-bold mb-2 ${getVerdictColor()}`}>
            {getVerdictText()}
          </h1>
          
          <div className="flex items-center justify-center gap-3 mb-4">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceLevelBadge()}`}>
              {prediction?.confidence_level}
            </span>
          </div>
          
          <p className="text-2xl text-gray-600 dark:text-gray-400 mb-4">
            Confidence: {prediction?.confidence.toFixed(1)}%
          </p>
          
          <div className="inline-block px-6 py-3 rounded-lg bg-gray-100 dark:bg-gray-700">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Fake Score: {prediction?.fake_score.toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Analysis Details
          </h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-500">Video ID</dt>
              <dd className="text-gray-900 dark:text-white font-mono text-sm">
                {prediction?.video_id}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Confidence Level</dt>
              <dd className={`font-medium ${prediction?.confidence_level === 'HIGH CONFIDENCE' ? 'text-success-600' : prediction?.confidence_level === 'REVIEW NEEDED' ? 'text-orange-600' : 'text-yellow-600'}`}>
                {prediction?.confidence_level}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Aggregation Method</dt>
              <dd className="text-gray-900 dark:text-white">
                {prediction?.aggregation_method}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Frames Analyzed</dt>
              <dd className="text-gray-900 dark:text-white">
                {prediction?.frame_predictions?.length || 0}
              </dd>
            </div>
          </dl>
        </div>
        
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Frame-by-Frame Results
          </h3>
          <div className="max-h-48 overflow-y-auto space-y-2">
            {prediction?.frame_predictions?.slice(0, 10).map((frame, idx) => (
              <div
                key={idx}
                className={`flex items-center justify-between p-2 rounded ${
                  frame.label === 'FAKE' ? 'bg-danger-100 dark:bg-danger-900/20' : 'bg-success-100 dark:bg-success-900/20'
                }`}
              >
                <span className="text-sm text-gray-600">
                  Frame {frame.frame_idx} ({frame.timestamp.toFixed(1)}s)
                </span>
                <span className={`text-sm font-medium ${
                  frame.label === 'FAKE' ? 'text-danger-600' : 'text-success-600'
                }`}>
                  {frame.label}
                </span>
              </div>
            ))}
          </div>
          {(prediction?.frame_predictions?.length || 0) > 10 && (
            <p className="text-sm text-gray-500 mt-2">
              And {(prediction?.frame_predictions?.length || 0) - 10} more frames...
            </p>
          )}
        </div>
      </div>
      
      {data?.audio_visual && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Audio-Visual Analysis
          </h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-500">Sync Score</dt>
              <dd className="text-gray-900 dark:text-white">
                {(data.audio_visual.sync_score * 100).toFixed(1)}%
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Lip Sync Status</dt>
              <dd className={data.audio_visual.is_desynced ? 'text-danger-600' : 'text-success-600'}>
                {data.audio_visual.is_desynced ? 'Desynchronized' : 'Synchronized'}
              </dd>
            </div>
          </dl>
        </div>
      )}
<<<<<<< HEAD

      {data?.gradcam_visualizations && data.gradcam_visualizations.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            GradCAM Heatmaps
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.gradcam_visualizations.slice(0, 6).map((cam, idx) => (
              <div key={idx} className="border rounded-lg overflow-hidden">
                <img
                  src={cam.overlay_url}
                  alt={`GradCAM heatmap for frame ${cam.frame_idx}`}
                  className="w-full h-auto"
                />
                <div className="p-2 text-sm text-gray-600 dark:text-gray-400">
                  Frame {cam.frame_idx} ({cam.timestamp.toFixed(1)}s)
                </div>
              </div>
            ))}
          </div>
          {data.gradcam_visualizations.length > 6 && (
            <p className="text-sm text-gray-500 mt-2">
              And {data.gradcam_visualizations.length - 6} more heatmaps...
            </p>
          )}
        </div>
      )}
=======
>>>>>>> 65700e59945d1b65257bc1d543bb8839aa765b7b
    </div>
  )
}

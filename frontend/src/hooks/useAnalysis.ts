import { useState, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { uploadVideo, getAnalysisResult } from '@/services/api'
import type { AnalysisResponse } from '@/types'

export function useAnalysis() {
  const [jobId, setJobId] = useState<string | null>(null)
  
  const uploadMutation = useMutation({
    mutationFn: uploadVideo,
    onSuccess: (data) => {
      setJobId(data.job_id)
    },
  })
  
  const resultQuery = useQuery({
    queryKey: ['analysis', jobId],
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
  
  const upload = useCallback(async (file: File) => {
    return uploadMutation.mutateAsync(file)
  }, [uploadMutation])
  
  return {
    upload,
    jobId,
    result: resultQuery.data as AnalysisResponse | undefined,
    status: resultQuery.data?.status,
    isLoading: uploadMutation.isPending || resultQuery.isLoading,
    isError: uploadMutation.isError || resultQuery.isError,
    error: uploadMutation.error || resultQuery.error,
    progress: uploadMutation.variables ? 100 : 0,
  }
}

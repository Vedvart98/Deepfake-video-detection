export interface VideoPrediction {
  video_id: string
  label: 'REAL' | 'FAKE' | 'UNCERTAIN'
  confidence: number
  confidence_level: 'HIGH CONFIDENCE' | 'LOW CONFIDENCE' | 'REVIEW NEEDED'
  fake_score: number
  aggregation_method: string
  frame_predictions: FramePrediction[]
}

export interface FramePrediction {
  frame_idx: number
  timestamp: number
  label: string
  confidence: number
  face_boxes: BoundingBox[]
}

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
  width: number
  height: number
}

export interface AudioVisual {
  sync_score: number
  is_desynced: boolean
  confidence: number
  temporal_alignment: TemporalAlignment[]
}

export interface TemporalAlignment {
  timestamp: number
  correlation: number
  frame_idx: number
}

export interface AnalysisResponse {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  video_id?: string
  video_prediction?: VideoPrediction
  audio_visual?: AudioVisual
  gradcam_visualizations?: GradCAM[]
  processing_time?: number
  error_message?: string
}

export interface GradCAM {
  frame_idx: number
  timestamp: number
  heatmap_url: string
  overlay_url: string
  manipulation_regions: ManipulationRegion[]
}

export interface ManipulationRegion {
  x: number
  y: number
  width: number
  height: number
}

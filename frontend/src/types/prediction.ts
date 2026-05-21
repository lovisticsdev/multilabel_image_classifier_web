export interface ImageInfo {
  filename: string;
  content_type: string;
  size_bytes: number;
  width: number;
  height: number;
}

export interface ClassPrediction {
  class_name: string;
  probability: number;
  threshold: number;
  is_predicted: boolean;
}

export interface PredictionResponse {
  image: ImageInfo;
  predicted_labels: ClassPrediction[];
  all_scores: ClassPrediction[];
  top_k: ClassPrediction[];
  model: Record<string, unknown>;
  message: string | null;
}

export interface ModelInfoResponse {
  available: boolean;
  architecture: string | null;
  num_classes: number | null;
  classes: string[];
  image_size: number | null;
  thresholds: Record<string, number>;
  validation_metrics: Record<string, unknown>;
  message: string | null;
}

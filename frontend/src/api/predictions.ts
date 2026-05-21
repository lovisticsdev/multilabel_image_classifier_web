import { requestJson } from './client';
import type { ModelInfoResponse, PredictionResponse } from '../types/prediction';

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  return requestJson<ModelInfoResponse>('/api/model');
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  return requestJson<PredictionResponse>('/api/predict', {
    method: 'POST',
    body: formData,
  });
}

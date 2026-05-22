import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, BrainCircuit } from 'lucide-react';
import { fetchModelInfo, predictImage } from '../api/predictions';
import { ModelInfoPanel } from '../components/ModelInfoPanel';
import { PredictionResults } from '../components/PredictionResults';
import { UploadDropzone } from '../components/UploadDropzone';
import type { ModelInfoResponse, PredictionResponse } from '../types/prediction';

export function HomePage() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);

  const loadModelInfo = useCallback(async () => {
    setIsLoadingModel(true);
    setModelError(null);
    try {
      setModelInfo(await fetchModelInfo());
    } catch (error) {
      setModelInfo(null);
      setModelError(error instanceof Error ? error.message : 'Could not load model information.');
    } finally {
      setIsLoadingModel(false);
    }
  }, []);

  useEffect(() => {
    void loadModelInfo();
  }, [loadModelInfo]);

  async function handlePredict(file: File) {
    setIsPredicting(true);
    try {
      const result = await predictImage(file);
      setPrediction(result);
    } finally {
      setIsPredicting(false);
    }
  }

  const uploadDisabledReason = useMemo(() => {
    if (isLoadingModel) return 'Checking the local model artifact.';
    if (modelError) return modelError;
    if (modelInfo?.available === false) return modelInfo.message ?? 'No exported model artifact is available.';
    return undefined;
  }, [isLoadingModel, modelError, modelInfo]);

  const statusLabel = modelInfo?.available ? 'Model ready' : isLoadingModel ? 'Checking model' : 'Model unavailable';

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><BrainCircuit aria-hidden="true" /></span>
          <div>
            <p className="eyebrow">VOC20 inference</p>
            <h1>Multi-label image classifier</h1>
          </div>
        </div>
        <div className={`status-badge${modelInfo?.available ? ' is-ready' : ''}`}>
          <Activity aria-hidden="true" />
          <span>{statusLabel}</span>
        </div>
      </header>

      <section className="workspace-grid">
        <div className="left-column">
          <UploadDropzone
            onPredict={handlePredict}
            disabled={isPredicting || Boolean(uploadDisabledReason)}
            disabledReason={uploadDisabledReason}
          />
          <ModelInfoPanel
            info={modelInfo}
            isLoading={isLoadingModel}
            error={modelError}
            onRefresh={loadModelInfo}
          />
        </div>
        <PredictionResults result={prediction} isLoading={isPredicting} />
      </section>
    </main>
  );
}

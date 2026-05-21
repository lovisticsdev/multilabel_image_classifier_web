import { useEffect, useState } from 'react';
import { Tags } from 'lucide-react';
import { fetchModelInfo, predictImage } from '../api/predictions';
import { ModelInfoPanel } from '../components/ModelInfoPanel';
import { PredictionResults } from '../components/PredictionResults';
import { UploadDropzone } from '../components/UploadDropzone';
import type { ModelInfoResponse, PredictionResponse } from '../types/prediction';

export function HomePage() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    fetchModelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  async function handlePredict(file: File) {
    setIsPredicting(true);
    try {
      const result = await predictImage(file);
      setPrediction(result);
    } finally {
      setIsPredicting(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-section">
        <div>
          <p className="eyebrow">Multi-label object tagging</p>
          <h1>Upload an image and get predicted object labels.</h1>
          <p>
            The API validates the upload, applies the same preprocessing used during training, and returns all
            configured class probabilities with threshold-based predictions.
          </p>
        </div>
        <div className="hero-icon"><Tags aria-hidden="true" /></div>
      </section>
      <div className="content-grid">
        <div className="stack">
          <UploadDropzone onPredict={handlePredict} disabled={isPredicting || modelInfo?.available === false} />
          <ModelInfoPanel info={modelInfo} />
        </div>
        <PredictionResults result={prediction} />
      </div>
    </main>
  );
}

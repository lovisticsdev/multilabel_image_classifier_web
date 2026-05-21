import { CheckCircle2, Info } from 'lucide-react';
import { ConfidenceBars } from './ConfidenceBars';
import type { PredictionResponse } from '../types/prediction';

interface PredictionResultsProps {
  result: PredictionResponse | null;
}

export function PredictionResults({ result }: PredictionResultsProps) {
  if (!result) {
    return (
      <section className="panel empty-panel">
        <Info aria-hidden="true" />
        <p>Upload an image to see predicted labels and all class probabilities.</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Results</p>
          <h2>Predicted labels</h2>
        </div>
        <CheckCircle2 aria-hidden="true" />
      </div>
      {result.predicted_labels.length > 0 ? (
        <div className="chip-list">
          {result.predicted_labels.map((item) => (
            <span className="chip" key={item.class_name}>{item.class_name} {(item.probability * 100).toFixed(1)}%</span>
          ))}
        </div>
      ) : (
        <p className="notice">{result.message ?? 'No confident label was predicted.'}</p>
      )}
      <h3>All class scores</h3>
      <ConfidenceBars scores={result.all_scores} />
    </section>
  );
}

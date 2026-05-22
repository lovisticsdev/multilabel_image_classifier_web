import { useState } from 'react';
import { CheckCircle2, ChevronDown, Info, Loader2, ScanSearch } from 'lucide-react';
import { ConfidenceBars } from './ConfidenceBars';
import type { PredictionResponse } from '../types/prediction';
import { formatLabel, formatPercent } from '../utils/labels';

interface PredictionResultsProps {
  result: PredictionResponse | null;
  isLoading: boolean;
}

function imageSize(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export function PredictionResults({ result, isLoading }: PredictionResultsProps) {
  const [showAllClasses, setShowAllClasses] = useState(false);

  if (isLoading) {
    return (
      <section className="tool-panel results-panel empty-panel">
        <Loader2 className="spin" aria-hidden="true" />
        <p>Analyzing image...</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="tool-panel results-panel empty-panel">
        <ScanSearch aria-hidden="true" />
        <p>Results will appear here.</p>
      </section>
    );
  }

  const topScore = result.top_k[0];

  return (
    <section className="tool-panel results-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Results</p>
          <h2>{result.image.filename}</h2>
        </div>
        <CheckCircle2 aria-hidden="true" />
      </div>

      <div className="summary-strip">
        <div><span>Predicted</span><strong>{result.predicted_labels.length}</strong></div>
        <div><span>Top score</span><strong>{topScore ? formatLabel(topScore.class_name) : 'n/a'}</strong></div>
        <div><span>Image</span><strong>{result.image.width}x{result.image.height}</strong></div>
        <div><span>Size</span><strong>{imageSize(result.image.size_bytes)}</strong></div>
      </div>

      {result.predicted_labels.length > 0 ? (
        <div className="chip-list">
          {result.predicted_labels.map((item) => (
            <span className="chip" key={item.class_name}>
              {formatLabel(item.class_name)} {formatPercent(item.probability)}
            </span>
          ))}
        </div>
      ) : (
        <p className="notice"><Info aria-hidden="true" />{result.message ?? 'No confident label was predicted.'}</p>
      )}

      <h3>Top scores</h3>
      <div className="top-score-list">
        {result.top_k.map((score, index) => (
          <div key={score.class_name}>
            <span>{index + 1}</span>
            <strong>{formatLabel(score.class_name)}</strong>
            <small>{formatPercent(score.probability)}</small>
          </div>
        ))}
      </div>

      <button
        className="section-toggle"
        type="button"
        onClick={() => setShowAllClasses((value) => !value)}
        aria-expanded={showAllClasses}
        aria-controls="all-class-scores"
      >
        <span>All classes</span>
        <small>{result.all_scores.length} scores</small>
        <ChevronDown aria-hidden="true" />
      </button>
      {showAllClasses && (
        <div id="all-class-scores" className="collapsible-region">
          <ConfidenceBars scores={result.all_scores} />
        </div>
      )}
    </section>
  );
}

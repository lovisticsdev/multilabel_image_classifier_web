import type { CSSProperties } from 'react';
import type { ClassPrediction } from '../types/prediction';
import { formatLabel, formatPercent } from '../utils/labels';

interface ConfidenceBarsProps {
  scores: ClassPrediction[];
}

export function ConfidenceBars({ scores }: ConfidenceBarsProps) {
  return (
    <div className="confidence-list">
      {scores.map((score) => {
        const probability = Math.max(0, Math.min(100, score.probability * 100));
        const threshold = Math.max(0, Math.min(100, score.threshold * 100));
        const style = {
          '--score-probability': `${probability}%`,
          '--score-threshold': `${threshold}%`,
        } as CSSProperties;

        return (
        <article
          className={score.is_predicted ? 'score-row is-predicted' : 'score-row'}
          key={score.class_name}
          style={style}
        >
          <div className="score-label">
            <strong>{formatLabel(score.class_name)}</strong>
            <span>{formatPercent(score.probability)}</span>
          </div>
          <div
            className="score-track"
            role="progressbar"
            aria-label={`${formatLabel(score.class_name)} probability`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(probability)}
          >
            <div className="score-fill" />
            <span className="threshold-marker" />
          </div>
          <small>Threshold {formatPercent(score.threshold, 0)}</small>
        </article>
        );
      })}
    </div>
  );
}

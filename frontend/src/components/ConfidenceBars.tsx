import type { ClassPrediction } from '../types/prediction';

interface ConfidenceBarsProps {
  scores: ClassPrediction[];
}

export function ConfidenceBars({ scores }: ConfidenceBarsProps) {
  return (
    <div className="confidence-list">
      {scores.map((score) => (
        <article className={score.is_predicted ? 'score-row predicted' : 'score-row'} key={score.class_name}>
          <div className="score-label">
            <strong>{score.class_name}</strong>
            <span>{(score.probability * 100).toFixed(1)}%</span>
          </div>
          <div className="score-track" aria-label={`${score.class_name} probability`}>
            <div style={{ width: `${Math.max(0, Math.min(100, score.probability * 100))}%` }} />
          </div>
          <small>threshold {(score.threshold * 100).toFixed(0)}%</small>
        </article>
      ))}
    </div>
  );
}

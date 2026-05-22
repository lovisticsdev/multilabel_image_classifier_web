import { useState } from 'react';
import { ChevronDown, Cpu, RefreshCcw } from 'lucide-react';
import type { ModelInfoResponse } from '../types/prediction';
import { formatLabel, formatPercent } from '../utils/labels';

interface ModelInfoPanelProps {
  info: ModelInfoResponse | null;
  isLoading: boolean;
  error: string | null;
  onRefresh: () => void;
}

function metricValue(metrics: Record<string, unknown>, key: string): string {
  const value = metrics[key];
  return typeof value === 'number' ? formatPercent(value) : 'n/a';
}

function trainedAt(value: string | null): string {
  if (!value) return 'n/a';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

export function ModelInfoPanel({ info, isLoading, error, onRefresh }: ModelInfoPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const forceExpanded = isLoading || Boolean(error) || info?.available === false;
  const showBody = forceExpanded || isExpanded;
  const canCollapse = info?.available === true && !isLoading && !error;

  return (
    <section className="tool-panel model-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Model</p>
          <h2>Inference artifact</h2>
        </div>
        <div className="panel-actions">
          {canCollapse && (
            <button
              className="collapse-button"
              type="button"
              onClick={() => setIsExpanded((value) => !value)}
              aria-expanded={showBody}
              aria-controls="model-card-details"
            >
              <span>{showBody ? 'Hide' : 'Show'}</span>
              <ChevronDown aria-hidden="true" />
            </button>
          )}
          <button className="icon-button" type="button" onClick={onRefresh} aria-label="Refresh model status">
            <RefreshCcw aria-hidden="true" />
          </button>
        </div>
      </div>

      {info?.available && (
        <div className="model-status-row">
          <Cpu aria-hidden="true" />
          <span>{info.architecture} · {info.num_classes} classes</span>
        </div>
      )}

      {showBody && (
        <div id="model-card-details" className="collapsible-region">
          {isLoading ? <p className="muted-text">Checking artifact...</p> : error ? (
            <p className="notice">{error}</p>
          ) : info?.available ? (
            <>
              <dl className="metadata-grid">
                <div><dt>Architecture</dt><dd>{info.architecture}</dd></div>
                <div><dt>Classes</dt><dd>{info.num_classes}</dd></div>
                <div><dt>Input</dt><dd>{info.image_size}px square</dd></div>
                <div><dt>Macro F1</dt><dd>{metricValue(info.validation_metrics, 'macro_f1')}</dd></div>
                <div><dt>Micro F1</dt><dd>{metricValue(info.validation_metrics, 'micro_f1')}</dd></div>
                <div><dt>Trained</dt><dd>{trainedAt(info.trained_at)}</dd></div>
              </dl>
              <div className="class-list" aria-label="Configured classes">
                {info.classes.slice(0, 20).map((className) => (
                  <span key={className}>{formatLabel(className)}</span>
                ))}
              </div>
            </>
          ) : (
            <p className="notice">{info?.message ?? 'No exported model artifact is available.'}</p>
          )}
        </div>
      )}
    </section>
  );
}

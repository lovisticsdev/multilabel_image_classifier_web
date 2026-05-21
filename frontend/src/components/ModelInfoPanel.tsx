import { Cpu } from 'lucide-react';
import type { ModelInfoResponse } from '../types/prediction';

interface ModelInfoPanelProps {
  info: ModelInfoResponse | null;
}

export function ModelInfoPanel({ info }: ModelInfoPanelProps) {
  return (
    <section className="panel model-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Model</p>
          <h2>Inference artifact</h2>
        </div>
        <Cpu aria-hidden="true" />
      </div>
      {!info ? <p>Loading model info...</p> : info.available ? (
        <dl className="metadata-grid">
          <div><dt>Architecture</dt><dd>{info.architecture}</dd></div>
          <div><dt>Classes</dt><dd>{info.num_classes}</dd></div>
          <div><dt>Image size</dt><dd>{info.image_size}px</dd></div>
        </dl>
      ) : (
        <p className="notice">{info.message}</p>
      )}
    </section>
  );
}

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react';
import { ImageUp, Loader2, UploadCloud } from 'lucide-react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_UPLOAD_MB = 10;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

interface UploadDropzoneProps {
  onPredict: (file: File) => Promise<void>;
  disabled?: boolean;
}

export function UploadDropzone({ onPredict, disabled = false }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  function validate(candidate: File): string | null {
    if (!ACCEPTED_TYPES.includes(candidate.type)) return 'Upload a JPEG, PNG, or WebP image.';
    if (candidate.size > MAX_UPLOAD_BYTES) return `Image must be ${MAX_UPLOAD_MB} MB or smaller.`;
    return null;
  }

  function setSelected(candidate: File) {
    const validation = validate(candidate);
    if (validation) {
      setError(validation);
      setFile(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
      return;
    }
    setError(null);
    setFile(candidate);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(candidate));
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) setSelected(selected);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) setSelected(dropped);
  }

  async function submit() {
    if (!file || isSubmitting || disabled) return;
    setIsSubmitting(true);
    setError(null);
    try {
      await onPredict(file);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="panel upload-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Predict</p>
          <h2>Upload an image</h2>
        </div>
        <ImageUp aria-hidden="true" />
      </div>
      <div
        className={`dropzone${isDragging ? ' is-dragging' : ''}`}
        onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleChange} />
        {previewUrl ? <img className="preview-image" src={previewUrl} alt="Selected upload preview" /> : (
          <div className="dropzone-placeholder">
            <UploadCloud aria-hidden="true" />
            <p>Drop an image here or click to browse.</p>
            <span>JPEG, PNG, or WebP. Max {MAX_UPLOAD_MB} MB.</span>
          </div>
        )}
      </div>
      {file && <p className="file-note">{file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB</p>}
      {error && <p className="error-message">{error}</p>}
      <button className="primary-button" onClick={submit} disabled={!file || isSubmitting || disabled}>
        {isSubmitting ? <Loader2 className="spin" aria-hidden="true" /> : null}
        {isSubmitting ? 'Predicting...' : 'Predict labels'}
      </button>
    </section>
  );
}

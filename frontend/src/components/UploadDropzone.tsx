import { ChangeEvent, DragEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { FileImage, ImageUp, Loader2, UploadCloud, X } from 'lucide-react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_UPLOAD_MB = 10;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

interface UploadDropzoneProps {
  onPredict: (file: File) => Promise<void>;
  disabled?: boolean;
  disabledReason?: string;
}

export function UploadDropzone({ onPredict, disabled = false, disabledReason }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const canInteract = !disabled && !isSubmitting;

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

  function clearSelection() {
    setFile(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = '';
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) setSelected(selected);
  }

  function openFilePicker() {
    if (canInteract) inputRef.current?.click();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    openFilePicker();
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    if (!canInteract) return;
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
    <section className="tool-panel upload-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Predict</p>
          <h2>Image input</h2>
        </div>
        <ImageUp aria-hidden="true" />
      </div>
      <div
        className={`dropzone${isDragging ? ' is-dragging' : ''}${disabled ? ' is-disabled' : ''}`}
        onDragOver={(event) => { event.preventDefault(); if (canInteract) setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={openFilePicker}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleChange}
          disabled={!canInteract}
        />
        {previewUrl ? <img className="preview-image" src={previewUrl} alt="Selected upload preview" /> : (
          <div className="dropzone-placeholder">
            <UploadCloud aria-hidden="true" />
            <p>Drop image or browse</p>
            <span>JPEG, PNG, or WebP. Max {MAX_UPLOAD_MB} MB.</span>
          </div>
        )}
      </div>
      {file && (
        <div className="file-row">
          <FileImage aria-hidden="true" />
          <span>{file.name}</span>
          <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
          <button className="icon-button" type="button" onClick={clearSelection} aria-label="Remove selected image">
            <X aria-hidden="true" />
          </button>
        </div>
      )}
      {disabledReason && <p className="notice">{disabledReason}</p>}
      {error && <p className="error-message" role="alert">{error}</p>}
      <button className="primary-button" onClick={submit} disabled={!file || isSubmitting || disabled}>
        {isSubmitting ? <Loader2 className="spin" aria-hidden="true" /> : null}
        {isSubmitting ? 'Analyzing...' : 'Analyze image'}
      </button>
    </section>
  );
}

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly details?: unknown;

  constructor(message: string, status: number, code?: string, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

function parseErrorBody(body: unknown): { message: string; code?: string; details?: unknown } {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string') return { message: detail, details: body };
    if (detail && typeof detail === 'object') {
      const maybe = detail as { code?: unknown; message?: unknown };
      return {
        message: typeof maybe.message === 'string' ? maybe.message : 'Request failed.',
        code: typeof maybe.code === 'string' ? maybe.code : undefined,
        details: body,
      };
    }
  }
  return { message: 'Request failed.', details: body };
}

export async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const contentType = response.headers.get('content-type') ?? '';
  const body = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const parsed = parseErrorBody(body);
    throw new ApiError(parsed.message, response.status, parsed.code, parsed.details);
  }
  return body as T;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(detail || `HTTP ${status}`);
    this.name = 'ApiError';
  }
}

/**
 * A route the backend doesn't have yet. FastAPI answers an unknown path with exactly
 * `{"detail": "Not Found"}` and a wrong method with 405, while a known route with an
 * unknown id carries its own message ("Run x not found"), which must stay a real 404.
 */
export function isUnavailable(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 405 || (err.status === 404 && err.detail === 'Not Found'));
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return String(err);
}

async function readDetail(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const body: unknown = JSON.parse(text);
    if (body && typeof body === 'object' && 'detail' in body) {
      const d = (body as { detail: unknown }).detail;
      if (typeof d === 'string') return d;
      if (Array.isArray(d)) {
        return d
          .map((e: unknown) => {
            if (e && typeof e === 'object' && 'msg' in e) {
              const loc = 'loc' in e && Array.isArray(e.loc) ? (e.loc as unknown[]).slice(1).join('.') : '';
              return loc ? `${loc}: ${String(e.msg)}` : String(e.msg);
            }
            return JSON.stringify(e);
          })
          .join('; ');
      }
      return JSON.stringify(d);
    }
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return text || res.statusText;
}

async function raw(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (err) {
    throw new ApiError(0, `Can't reach the server (${errorMessage(err)})`);
  }
  if (!res.ok) throw new ApiError(res.status, await readDetail(res));
  return res;
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await raw(path, { signal });
  return (await res.json()) as T;
}

export async function sendJson<T>(method: 'POST' | 'PUT' | 'DELETE', path: string, body?: unknown): Promise<T> {
  const res = await raw(path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export async function getBlob(path: string): Promise<{ blob: Blob; filename: string | null }> {
  const res = await raw(path);
  const cd = res.headers.get('Content-Disposition') ?? '';
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  return { blob: await res.blob(), filename: m ? decodeURIComponent(m[1]) : null };
}

/** Status of a probe request, without throwing. */
export async function probeStatus(path: string, method: 'GET' | 'POST' = 'GET', body?: unknown): Promise<{ status: number; detail: string }> {
  try {
    const res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.ok) return { status: res.status, detail: '' };
    return { status: res.status, detail: await readDetail(res) };
  } catch {
    return { status: 0, detail: '' };
  }
}

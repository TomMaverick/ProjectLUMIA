/**
 * Resilient HTTP client for Project LUMIA REST API.
 * Features:
 * - Automatic Authorization: Bearer <token> injection
 * - Credentials: "include" for HttpOnly refresh cookie rotation
 * - Single-flight refresh token mutex on 401 Unauthorized
 * - Strict JSON parsing & timeout handling
 */

import type { AccessTokenResponse } from "../types/api";

function getInitialApiBaseUrl(): string {
  try {
    const stored = globalThis.localStorage?.getItem("lumia.ui.settings.v1");
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed.backend_connection === "remote" && parsed.remote_base_url) {
        return parsed.remote_base_url.replace(/\/+$/, "");
      }
    }
  } catch {
    // Ignore storage errors in non-browser/SSR environments
  }
  return import.meta.env.VITE_API_BASE_URL ?? "";
}

let apiBaseUrl: string = getInitialApiBaseUrl();
const DEFAULT_TIMEOUT_MS = 8000;

export function setApiBaseUrl(url: string): void {
  apiBaseUrl = url ? url.replace(/\/+$/, "") : "";
}

export function getApiBaseUrl(): string {
  return apiBaseUrl;
}

let accessToken: string | null = null;
let refreshPromise: Promise<string | null> | null = null;

export class HttpClientError extends Error {
  public readonly statusCode: number | null;
  public readonly responseBody: unknown;

  public constructor(message: string, statusCode: number | null = null, responseBody: unknown = null) {
    super(message);
    this.name = "HttpClientError";
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export function getAccessToken(): string | null {
  return accessToken;
}

/**
 * Perform single-flight refresh token request using the HttpOnly lumia_refresh_token cookie.
 */
export async function refreshAccessToken(): Promise<string | null> {
  if (refreshPromise) {
    return refreshPromise;
  }

  refreshPromise = (async () => {
    try {
      const response = await globalThis.fetch(`${getApiBaseUrl()}/api/v1/auth/refresh`, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        setAccessToken(null);
        return null;
      }

      const data = (await response.json()) as AccessTokenResponse;
      setAccessToken(data.access_token);
      return data.access_token;
    } catch {
      setAccessToken(null);
      return null;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

export interface RequestOptions extends RequestInit {
  readonly timeoutMs?: number;
  readonly skipAuth?: boolean;
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, skipAuth = false, signal, ...restOptions } = options;

  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), timeoutMs);

  const onAbort = (): void => controller.abort();
  if (signal) {
    if (signal.aborted) {
      globalThis.clearTimeout(timeoutId);
      throw new HttpClientError("Request aborted", null);
    }
    signal.addEventListener("abort", onAbort, { once: true });
  }

  const headers = new Headers(restOptions.headers);
  headers.set("Accept", "application/json");

  if (!skipAuth && accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  if (restOptions.body && !(restOptions.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    let response = await globalThis.fetch(`${getApiBaseUrl()}${path}`, {
      ...restOptions,
      headers,
      credentials: "include",
      signal: controller.signal,
    });

    // Auto-refresh token if 401 Unauthorized encountered and not explicitly skipping auth
    if (response.status === 401 && !skipAuth) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        headers.set("Authorization", `Bearer ${newToken}`);
        response = await globalThis.fetch(`${getApiBaseUrl()}${path}`, {
          ...restOptions,
          headers,
          credentials: "include",
          signal: controller.signal,
        });
      }
    }

    if (!response.ok) {
      let body: unknown = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      throw new HttpClientError(`HTTP ${response.status} on ${path}`, response.status, body);
    }

    if (response.status === 204) {
      return null as T;
    }

    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof HttpClientError) {
      throw err;
    }
    const reason = err instanceof Error ? err.message : "Network error";
    throw new HttpClientError(`Request failed: ${reason}`);
  } finally {
    globalThis.clearTimeout(timeoutId);
    if (signal) {
      signal.removeEventListener("abort", onAbort);
    }
  }
}

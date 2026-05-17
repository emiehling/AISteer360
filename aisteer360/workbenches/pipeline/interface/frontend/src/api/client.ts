const TOKEN_KEY = "aisteer360-owner-token";

function getToken(): string {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    throw new Error("missing owner token");
  }
  return token;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${getToken()}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const resp = await fetch(path, { ...init, headers });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      // body is not JSON
    }
    throw new Error(`${path}: ${detail}`);
  }
  return (await resp.json()) as T;
}

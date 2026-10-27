// Sent on every request so localtunnel does not return its interstitial page.
export const API_HEADERS = { "bypass-tunnel-reminder": "true" };

let cached: Promise<string> | null = null;

async function loadApiBase(): Promise<string> {
  // Opened from this machine (the local demo): talk to the local back end, no tunnel needed.
  if (typeof window !== "undefined" && ["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    return process.env.NEXT_PUBLIC_API_URL || `http://${window.location.hostname}:8000`;
  }
  // public/config.json lets the backend address change with a commit instead of a rebuild.
  try {
    const res = await fetch("/config.json", { cache: "no-store" });
    if (res.ok) {
      const cfg = await res.json();
      if (typeof cfg.apiBase === "string" && /^https?:\/\//.test(cfg.apiBase)) {
        return cfg.apiBase.replace(/\/+$/, "");
      }
    }
  } catch {
    /* fall through to the build-time value */
  }
  return (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");
}

export function getApiBase(): Promise<string> {
  if (!cached) cached = loadApiBase();
  return cached;
}

// Free tunnels drop the odd request. Retry network failures and gateway errors a couple of times.
export async function fetchRetry(url: string, init?: RequestInit, tries = 3): Promise<Response> {
  let lastError: unknown = new Error("Request failed");
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      const res = await fetch(url, init);
      if (![502, 503, 504].includes(res.status)) return res;
      lastError = new Error(`Server returned ${res.status}`);
    } catch (e) {
      lastError = e;
    }
    await new Promise((resolve) => setTimeout(resolve, 600 * (attempt + 1)));
  }
  throw lastError;
}

// Sent on every request so localtunnel does not return its interstitial page.
export const API_HEADERS = { "bypass-tunnel-reminder": "true" };

let cached: Promise<string> | null = null;

async function loadApiBase(): Promise<string> {
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

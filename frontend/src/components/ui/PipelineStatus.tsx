"use client";

import { useEffect, useState } from "react";
import { API_HEADERS, getApiBase } from "@/lib/api";

type State = "checking" | "online" | "offline";

const POLL_MS = 15_000;

const LABEL: Record<State, string> = {
  checking: "Checking pipeline",
  online: "Pipeline Online",
  offline: "Pipeline Offline",
};

const DOT: Record<State, string> = {
  checking: "bg-stone-400 dark:bg-zinc-500",
  online: "bg-emerald-500",
  offline: "bg-rose-500",
};

// Asks the back end whether an inference worker is ready, instead of assuming it is.
export function PipelineStatus() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const check = async () => {
      let next: State = "offline";
      try {
        const base = await getApiBase();
        const res = await fetch(`${base}/api/health`, { headers: API_HEADERS, cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          next = data.worker ? "online" : "offline";
        }
      } catch {
        /* unreachable: stays offline */
      }
      if (cancelled) return;
      setState(next);
      timer = setTimeout(check, POLL_MS);
    };
    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return (
    <div
      role="status"
      className="flex items-center gap-2 px-2.5 py-1 rounded-full whitespace-nowrap bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800 text-xs font-mono text-stone-800 dark:text-zinc-300"
    >
      <span className={`w-2 h-2 rounded-full ${DOT[state]}`} aria-hidden="true"></span>
      <span>{LABEL[state]}</span>
    </div>
  );
}

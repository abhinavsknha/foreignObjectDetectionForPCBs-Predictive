"use client";

import { useLayoutEffect } from "react";
import { Moon, Sun } from "lucide-react";

type Theme = "light" | "dark";

function readStored(): Theme | null {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

function systemTheme(): Theme {
  try {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function ThemeToggle() {
  // React's dev-mode remount clears the attribute the inline script set; re-apply it before paint.
  useLayoutEffect(() => {
    document.documentElement.setAttribute("data-theme", readStored() ?? systemTheme());
  }, []);

  const toggle = () => {
    const current = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const next: Theme = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* storage unavailable: the choice still applies for this visit */
    }
  };

  // Both icons are always rendered and CSS shows the one for the target theme,
  // so server and client markup match and there is no flash.
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Switch between light and dark theme"
      title="Switch between light and dark theme"
      className="inline-flex shrink-0 items-center justify-center w-9 h-9 rounded-lg border border-stone-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-stone-700 dark:text-zinc-300 hover:bg-stone-100 dark:hover:bg-zinc-800 hover:text-stone-900 dark:hover:text-zinc-100 transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600 dark:focus-visible:outline-emerald-400"
    >
      <Moon className="w-4 h-4 dark:hidden" aria-hidden="true" />
      <Sun className="w-4 h-4 hidden dark:block" aria-hidden="true" />
    </button>
  );
}

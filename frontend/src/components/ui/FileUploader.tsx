"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, Loader2, Image as ImageIcon, Check, AlertTriangle, RotateCcw, Minus } from "lucide-react";
import { API_HEADERS, getApiBase } from "@/lib/api";
import type { InferenceResults, Route } from "@/lib/types";

interface FileUploaderProps {
  onInferenceComplete: (fileKey: string, results: InferenceResults) => void;
}

type Status = "idle" | "uploading" | "processing" | "error";

const MAX_BYTES = 25 * 1024 * 1024;
const POLL_MS = 400;
const STEP_MS = 450; // minimum time a stage stays on screen so each one can be read
const TIMEOUT_MS = 180_000;

// Backend stage -> index of the step that is active while it runs.
const STAGE_INDEX: Record<string, number> = {
  queued: 1,
  checking_resolution: 1,
  enhancing: 2,
  detecting: 3,
  fusing: 4,
};
const STEP_COUNT = 5;

function stepLabel(i: number, route: Route | null): { label: string; detail?: string } {
  switch (i) {
    case 0:
      return { label: "Upload scan" };
    case 1:
      return {
        label: "Check resolution",
        detail: route
          ? `${route.native.width} × ${route.native.height} px, edge, contrast and frequency tests`
          : "Edge density, contrast and frequency content",
      };
    case 2:
      if (route && !route.needs_sr) return { label: "No enhancement needed", detail: "Resolution is sufficient" };
      return {
        label: route ? "Enhance resolution" : "Enhance resolution if needed",
        detail: route?.method ?? undefined,
      };
    case 3:
      return { label: "Detect defects", detail: "YOLO26n and RT-DETR-L" };
    default:
      return { label: "Fuse boxes", detail: "Weighted Box Fusion" };
  }
}

export function FileUploader({ onInferenceComplete }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [route, setRoute] = useState<Route | null>(null);
  const [targetIdx, setTargetIdx] = useState(0);
  const [shownIdx, setShownIdx] = useState(0);
  const [finalResults, setFinalResults] = useState<InferenceResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [lastFile, setLastFile] = useState<File | null>(null);
  const onCompleteRef = useRef(onInferenceComplete);
  useEffect(() => {
    onCompleteRef.current = onInferenceComplete;
  }, [onInferenceComplete]);

  const fail = useCallback((message: string) => {
    setError(message);
    setStatus("error");
  }, []);

  // Poll the backend for the stage it is in and for the final result.
  useEffect(() => {
    if (status !== "processing" || !taskId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    const started = Date.now();

    const tick = async () => {
      if (cancelled) return;
      try {
        const base = await getApiBase();
        const res = await fetch(`${base}/api/inference/status/${taskId}`, { headers: API_HEADERS, cache: "no-store" });
        if (!res.ok) throw new Error(`Status check returned ${res.status}`);
        const data = await res.json();
        failures = 0;
        if (cancelled) return;
        if (data.route) setRoute(data.route);
        if (data.status === "completed") {
          const results: InferenceResults = data.result.results;
          if (results.route) setRoute(results.route);
          setFinalResults(results);
          setTargetIdx(STEP_COUNT);
          return;
        }
        if (data.status === "failed") {
          fail(data.error || "The inspection failed on the server.");
          return;
        }
        setTargetIdx((t) => Math.max(t, STAGE_INDEX[data.stage] ?? 1));
      } catch {
        failures += 1;
        if (failures >= 5) {
          fail("Lost connection to the inspection backend.");
          return;
        }
      }
      if (Date.now() - started > TIMEOUT_MS) {
        fail("The inspection is taking too long. The inference worker may be offline.");
        return;
      }
      timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [status, taskId, fail]);

  // Step the display forward one stage at a time.
  useEffect(() => {
    if (shownIdx >= targetIdx) return;
    const skipping = shownIdx + 1 === 2 && route !== null && !route.needs_sr;
    const t = setTimeout(() => setShownIdx((i) => i + 1), skipping ? 150 : STEP_MS);
    return () => clearTimeout(t);
  }, [shownIdx, targetIdx, route]);

  // Hand over to the results screen once the last stage has been shown.
  useEffect(() => {
    if (finalResults && shownIdx >= STEP_COUNT && previewUrl) {
      onCompleteRef.current(previewUrl, finalResults);
    }
  }, [finalResults, shownIdx, previewUrl]);

  const uploadFile = async (selectedFile: File) => {
    setLastFile(selectedFile);
    if (!selectedFile.type.startsWith("image/")) {
      fail("That file is not an image. Use a PNG, JPG, JPEG or BMP scan.");
      return;
    }
    if (selectedFile.size > MAX_BYTES) {
      fail("That image is larger than 25 MB.");
      return;
    }

    setError(null);
    setRoute(null);
    setFinalResults(null);
    setTargetIdx(0);
    setShownIdx(0);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setStatus("uploading");

    try {
      const base = await getApiBase();
      const formData = new FormData();
      formData.append("file", selectedFile);

      const uploadRes = await fetch(`${base}/api/upload/local`, { method: "POST", headers: API_HEADERS, body: formData });
      if (!uploadRes.ok) {
        const detail = await uploadRes.json().catch(() => null);
        throw new Error(detail?.detail || `Upload failed (${uploadRes.status})`);
      }
      const uploadData = await uploadRes.json();

      const inferRes = await fetch(`${base}/api/inference/trigger?file_key=${encodeURIComponent(uploadData.file_key)}`, {
        method: "POST",
        headers: API_HEADERS,
      });
      if (!inferRes.ok) throw new Error(`Could not start the inspection (${inferRes.status})`);
      const inferData = await inferRes.json();

      setTaskId(inferData.task_id);
      setTargetIdx(1);
      setStatus("processing");
    } catch (e) {
      const offline = e instanceof TypeError;
      fail(
        offline
          ? "Cannot reach the inspection backend. It may be offline; try again in a moment."
          : e instanceof Error
            ? e.message
            : "Upload failed."
      );
    }
  };

  const reset = () => {
    setStatus("idle");
    setError(null);
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    },
    // uploadFile only touches state setters and refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  return (
    <div className="w-full">
      <div
        className={`relative overflow-hidden rounded-2xl border-2 border-dashed transition-all p-10 sm:p-14 text-center flex flex-col items-center justify-center min-h-[340px] ${
          isDragging
            ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20"
            : "border-stone-300 dark:border-zinc-700/80 bg-[#fffdf8] dark:bg-zinc-900/40 hover:bg-white dark:hover:bg-zinc-900/60 hover:border-stone-400 dark:hover:border-zinc-600"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setIsDragging(false);
        }}
        onDrop={handleDrop}
      >
        <AnimatePresence mode="wait">
          {status === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
              className="flex flex-col items-center gap-4 max-w-md mx-auto"
            >
              <div className="w-14 h-14 rounded-2xl bg-stone-100 dark:bg-zinc-800/90 border border-stone-300 dark:border-zinc-700/80 flex items-center justify-center text-stone-800 dark:text-zinc-300 shadow-sm">
                <UploadCloud className="w-7 h-7 text-stone-800 dark:text-zinc-300" />
              </div>

              <div className="space-y-1">
                <h3 className="text-base sm:text-lg font-medium text-stone-900 dark:text-zinc-100">
                  Select or drop a PCB inspection scan
                </h3>
                <p className="text-xs text-stone-700 dark:text-zinc-400">
                  Drag and drop your board scan here, or browse files from your computer.
                </p>
              </div>

              <label className="mt-2 inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-stone-900 dark:bg-zinc-100 hover:bg-stone-800 dark:hover:bg-white text-[#faf8f3] dark:text-zinc-900 text-xs sm:text-sm font-medium cursor-pointer transition-colors shadow-sm focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-emerald-600 dark:focus-within:outline-emerald-400">
                <ImageIcon className="w-4 h-4 text-stone-300 dark:text-zinc-700" />
                <span>Browse Scan Image</span>
                <input
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  onChange={(e) => {
                    if (e.target.files?.length) uploadFile(e.target.files[0]);
                    e.target.value = "";
                  }}
                />
              </label>

              <p className="text-[11px] font-mono text-stone-600 dark:text-zinc-400 pt-1">
                Accepted: PNG, JPG, JPEG, BMP (up to 25 MB)
              </p>
            </motion.div>
          )}

          {status === "uploading" && (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
              className="flex flex-col items-center gap-4 max-w-sm mx-auto"
            >
              <div className="w-12 h-12 rounded-xl bg-stone-100 dark:bg-zinc-800 border border-stone-300 dark:border-zinc-700 flex items-center justify-center">
                <Loader2 className="w-6 h-6 animate-spin text-stone-800 dark:text-zinc-300" />
              </div>
              <div className="space-y-1 text-center">
                <h3 className="text-sm font-medium text-stone-900 dark:text-zinc-200">Uploading Scan</h3>
                <p className="text-xs font-mono text-stone-700 dark:text-zinc-400">Sending the image to the inspection backend</p>
              </div>
            </motion.div>
          )}

          {status === "processing" && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
              className="flex flex-col items-stretch gap-5 w-full max-w-sm mx-auto text-left"
              role="status"
              aria-live="polite"
            >
              <div className="text-center space-y-1">
                <h3 className="text-sm font-medium text-stone-900 dark:text-zinc-100">Inspecting scan</h3>
                <p className="text-xs font-mono text-stone-700 dark:text-zinc-400">
                  {shownIdx >= STEP_COUNT ? "Done" : stepLabel(Math.min(shownIdx, STEP_COUNT - 1), route).label}
                </p>
              </div>

              <ol className="space-y-2.5">
                {Array.from({ length: STEP_COUNT }, (_, i) => {
                  const skipped = i === 2 && route !== null && !route.needs_sr;
                  const done = i < shownIdx;
                  const active = i === shownIdx;
                  const { label, detail } = stepLabel(i, route);
                  return (
                    <li key={i} className="flex items-start gap-3">
                      <span
                        className={`mt-0.5 w-5 h-5 shrink-0 rounded-full flex items-center justify-center border ${
                          done
                            ? skipped
                              ? "bg-stone-100 dark:bg-zinc-800 border-stone-300 dark:border-zinc-700 text-stone-600 dark:text-zinc-400"
                              : "bg-emerald-600 dark:bg-emerald-500 border-emerald-600 dark:border-emerald-500 text-white dark:text-zinc-950"
                            : active
                              ? "border-emerald-600 dark:border-emerald-400 text-emerald-700 dark:text-emerald-400"
                              : "border-stone-300 dark:border-zinc-700 text-transparent"
                        }`}
                      >
                        {done && skipped && <Minus className="w-3 h-3" aria-hidden="true" />}
                        {done && !skipped && <Check className="w-3 h-3" aria-hidden="true" />}
                        {active && <Loader2 className="w-3 h-3 animate-spin" aria-hidden="true" />}
                      </span>
                      <span className="min-w-0">
                        <span
                          className={`block text-xs font-medium ${
                            done || active ? "text-stone-900 dark:text-zinc-100" : "text-stone-600 dark:text-zinc-400"
                          }`}
                        >
                          {label}
                        </span>
                        {(done || active) && detail && (
                          <span className="block text-[11px] font-mono text-stone-700 dark:text-zinc-400">{detail}</span>
                        )}
                      </span>
                    </li>
                  );
                })}
              </ol>
            </motion.div>
          )}

          {status === "error" && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
              className="flex flex-col items-center gap-4 max-w-sm mx-auto"
              role="alert"
            >
              <div className="w-12 h-12 rounded-xl bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/60 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-rose-700 dark:text-rose-300" aria-hidden="true" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-medium text-stone-900 dark:text-zinc-100">Inspection did not complete</h3>
                <p className="text-xs text-stone-700 dark:text-zinc-400">{error}</p>
              </div>
              <div className="flex items-center gap-2">
                {lastFile && (
                  <button
                    type="button"
                    onClick={() => uploadFile(lastFile)}
                    className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-stone-900 dark:bg-zinc-100 hover:bg-stone-800 dark:hover:bg-white text-[#faf8f3] dark:text-zinc-900 text-xs font-medium cursor-pointer transition-colors"
                  >
                    <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                    Try again
                  </button>
                )}
                <button
                  type="button"
                  onClick={reset}
                  className="px-3.5 py-1.5 rounded-lg border border-stone-300 dark:border-zinc-700 text-xs font-medium text-stone-800 dark:text-zinc-200 hover:bg-stone-100 dark:hover:bg-zinc-800 cursor-pointer transition-colors"
                >
                  Choose another image
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  Download,
  AlertTriangle,
  Cpu,
  SlidersHorizontal,
  CheckCircle2,
  Crosshair,
  Layers,
  ScanSearch,
} from "lucide-react";
import type { Detection, InferenceResults } from "@/lib/types";

interface ResultsViewProps {
  fileKey: string;
  results: InferenceResults;
  onReset: () => void;
}

const MAX_CANVAS_H = 560;
type HoverSource = "canvas" | "list";

const csvCell = (v: string | number | null | undefined) => {
  const text = v == null ? "" : String(v);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

// One row per fused detection currently shown, in the pixels of the uploaded image.
function detectionsToCsv(detections: Detection[]): string {
  const header = ["class", "defect", "confidence", "x1", "y1", "x2", "y2", "source", "yolo26n_confidence", "rtdetr_confidence"];
  const rows = detections.map((d) => [
    d.class,
    d.full_name,
    d.confidence.toFixed(4),
    ...d.bbox.map((n) => n.toFixed(1)),
    d.source,
    d.yolo_conf == null ? "" : d.yolo_conf.toFixed(4),
    d.detr_conf == null ? "" : d.detr_conf.toFixed(4),
  ]);
  return [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n") + "\r\n";
}

export function ResultsView({ fileKey, results, onReset }: ResultsViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  // 0.25 is the reporting threshold the detectors were evaluated at.
  const [minConf, setMinConf] = useState(0.25);
  const [hovered, setHovered] = useState<{ index: number; source: HoverSource } | null>(null);
  const [loaded, setLoaded] = useState<{ src: string; el: HTMLImageElement } | null>(null);
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const [containerW, setContainerW] = useState(0);

  const hoveredIndex = hovered?.index ?? null;
  const filteredDetections = useMemo(
    () => results.detections.filter((d) => d.confidence >= minConf),
    [results.detections, minConf]
  );

  // Load the uploaded image once; hover redraws must not reload it.
  useEffect(() => {
    const el = new Image();
    el.onload = () => setLoaded({ src: fileKey, el });
    el.onerror = () => setFailedSrc(fileKey);
    el.src = fileKey;
  }, [fileKey]);
  const img = loaded?.src === fileKey ? loaded.el : null;
  const imgFailed = failedSrc === fileKey;

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const update = () => setContainerW(node.clientWidth);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // Displayed size: fit the image inside the pane without distorting it.
  const layout = useMemo(() => {
    if (!img || containerW === 0) return null;
    const scale = Math.min(containerW / img.naturalWidth, MAX_CANVAS_H / img.naturalHeight);
    return { scale, cssW: img.naturalWidth * scale, cssH: img.naturalHeight * scale };
  }, [img, containerW]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { scale, cssW, cssH } = layout;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.drawImage(img, 0, 0, cssW, cssH);

    filteredDetections.forEach((det, i) => {
      const isHovered = hoveredIndex === i;
      const [x1, y1, x2, y2] = det.bbox;
      const bx = x1 * scale;
      const by = y1 * scale;
      const bw = (x2 - x1) * scale;
      const bh = (y2 - y1) * scale;

      if (isHovered) {
        ctx.fillStyle = "rgba(56, 189, 248, 0.18)";
        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(bx, by, bw, bh);

        // Crosshair corner guides
        const c = Math.min(10, bw / 3, bh / 3);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(bx, by + c); ctx.lineTo(bx, by); ctx.lineTo(bx + c, by);
        ctx.moveTo(bx + bw - c, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + c);
        ctx.moveTo(bx, by + bh - c); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + c, by + bh);
        ctx.moveTo(bx + bw - c, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - c);
        ctx.stroke();
      } else {
        ctx.strokeStyle = "#f43f5e";
        ctx.lineWidth = 1.75;
        ctx.strokeRect(bx, by, bw, bh);
      }

      // Label pill, kept inside the canvas
      const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "600 11px monospace";
      const pillW = ctx.measureText(label).width + 10;
      const pillH = 18;
      const pillX = Math.min(Math.max(0, bx), Math.max(0, cssW - pillW));
      const pillY = by - pillH - 3 >= 0 ? by - pillH - 3 : Math.min(by + 2, cssH - pillH);

      ctx.fillStyle = isHovered ? "#0284c7" : "rgba(15, 23, 42, 0.9)";
      ctx.fillRect(pillX, pillY, pillW, pillH);
      ctx.strokeStyle = isHovered ? "#38bdf8" : "#f43f5e";
      ctx.lineWidth = 1;
      ctx.strokeRect(pillX, pillY, pillW, pillH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, pillX + 5, pillY + 13);
    });
  }, [img, layout, filteredDetections, hoveredIndex]);

  // Hovering a box on the image brings its catalogue entry into view (inside the list only).
  useEffect(() => {
    if (!hovered || hovered.source !== "canvas") return;
    const list = listRef.current;
    const item = itemRefs.current[hovered.index];
    if (!list || !item) return;
    const top = item.offsetTop - list.offsetTop;
    if (top < list.scrollTop) list.scrollTop = top - 4;
    else if (top + item.offsetHeight > list.scrollTop + list.clientHeight)
      list.scrollTop = top + item.offsetHeight - list.clientHeight + 4;
  }, [hovered]);

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!layout) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / layout.scale;
    const y = (e.clientY - rect.top) / layout.scale;

    // When boxes overlap, pick the smallest one under the pointer.
    let best = -1;
    let bestArea = Infinity;
    filteredDetections.forEach((det, i) => {
      const [x1, y1, x2, y2] = det.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        const area = (x2 - x1) * (y2 - y1);
        if (area < bestArea) {
          best = i;
          bestArea = area;
        }
      }
    });
    setHovered(best === -1 ? null : { index: best, source: "canvas" });
  };

  const activeDetection: Detection | null = hoveredIndex !== null ? filteredDetections[hoveredIndex] ?? null : null;

  // Detail card sits under the highlighted box, or above it when there is no room below.
  const tooltipStyle = useMemo<React.CSSProperties | null>(() => {
    if (!activeDetection || !layout) return null;
    const [x1, y1, , y2] = activeDetection.bbox;
    const cardW = Math.min(250, layout.cssW);
    const left = Math.min(Math.max(0, x1 * layout.scale), Math.max(0, layout.cssW - cardW));
    const below = y2 * layout.scale + 8;
    const top = below + 96 <= layout.cssH ? below : Math.max(0, y1 * layout.scale - 104);
    return { left, top, width: cardW };
  }, [activeDetection, layout]);

  const exportCsv = () => {
    const blob = new Blob([detectionsToCsv(filteredDetections)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    link.download = `pcb-inspection-${stamp}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const route = results.route;
  const imgW = img?.naturalWidth ?? results.image?.width;
  const imgH = img?.naturalHeight ?? results.image?.height;

  return (
    <div className="w-full bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800 rounded-2xl overflow-hidden shadow-xl text-stone-900 dark:text-zinc-100 flex flex-col">
      {/* Top Action Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 px-6 py-4 border-b border-stone-200 dark:border-zinc-800 bg-[#fffdf8] dark:bg-zinc-900/90">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-stone-100 dark:bg-zinc-800 border border-stone-300 dark:border-zinc-700 flex items-center justify-center text-stone-800 dark:text-zinc-300">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-stone-900 dark:text-zinc-100">Inspection Analysis</h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-stone-100 dark:bg-zinc-800 text-stone-800 dark:text-zinc-300 border border-stone-300 dark:border-zinc-700">
                WBF Ensemble Fused
              </span>
            </div>
            <p className="text-xs text-stone-700 dark:text-zinc-400 font-mono">
              {filteredDetections.length} anomaly targets identified
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={exportCsv}
            disabled={filteredDetections.length === 0}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-stone-100 dark:bg-zinc-800 hover:bg-stone-200 dark:hover:bg-zinc-700 border border-stone-300 dark:border-zinc-700 text-xs font-medium text-stone-900 dark:text-zinc-200 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-stone-100 dark:bg-zinc-800 hover:bg-stone-200 dark:hover:bg-zinc-700 border border-stone-300 dark:border-zinc-700 text-xs font-medium text-stone-900 dark:text-zinc-200 transition-colors cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Inspect Next Scan</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-stone-200 dark:divide-zinc-800">
        {/* Left Column: Canvas Viewport & Controls */}
        <div className="lg:col-span-7 p-6 bg-[#f7f4ec] dark:bg-zinc-950 flex flex-col justify-between">
          <div className="space-y-4">
            {/* Filter Slider */}
            <div className="flex items-center justify-between gap-4 p-3 rounded-xl bg-[#fffdf8] dark:bg-zinc-900/60 border border-stone-200 dark:border-zinc-800">
              <label htmlFor="conf-slider" className="flex items-center gap-2 text-xs font-medium text-stone-800 dark:text-zinc-300">
                <SlidersHorizontal className="w-4 h-4 text-stone-700 dark:text-zinc-400" />
                <span>Confidence Threshold</span>
              </label>

              <div className="flex items-center gap-3 flex-1 max-w-xs">
                <input
                  id="conf-slider"
                  type="range"
                  min="20"
                  max="95"
                  value={Math.round(minConf * 100)}
                  onChange={(e) => {
                    setHovered(null);
                    setMinConf(parseInt(e.target.value) / 100);
                  }}
                  className="w-full h-1.5 bg-stone-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-600 dark:accent-emerald-500"
                />
                <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-stone-100 dark:bg-zinc-800 border border-stone-300 dark:border-zinc-700 text-stone-900 dark:text-zinc-200 min-w-[42px] text-center">
                  {(minConf * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Inspection Canvas Area */}
            <div
              ref={containerRef}
              className="w-full rounded-xl overflow-hidden border border-stone-200 dark:border-zinc-800 bg-white dark:bg-zinc-900"
              style={{ minHeight: layout ? undefined : 240 }}
            >
              {imgFailed ? (
                <p className="p-6 text-xs text-center text-stone-700 dark:text-zinc-400">
                  The uploaded image could not be displayed.
                </p>
              ) : (
                <div className="relative mx-auto" style={layout ? { width: layout.cssW, height: layout.cssH } : undefined}>
                  <canvas
                    ref={canvasRef}
                    role="img"
                    aria-label={`Uploaded scan with ${filteredDetections.length} detected defects outlined`}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseLeave={() => setHovered((h) => (h?.source === "canvas" ? null : h))}
                    className="cursor-crosshair block"
                    style={layout ? { width: layout.cssW, height: layout.cssH } : undefined}
                  />
                  {activeDetection && tooltipStyle && (
                    <div
                      className="pointer-events-none absolute z-10 rounded-lg border border-sky-500 bg-white/95 dark:bg-zinc-900/95 p-2.5 text-[11px] shadow-lg text-stone-900 dark:text-zinc-100"
                      style={tooltipStyle}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold truncate">{activeDetection.full_name}</span>
                        <span className="font-mono shrink-0">{(activeDetection.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="mt-1 font-mono text-stone-700 dark:text-zinc-400">
                        {activeDetection.source === "ENSEMBLE (WBF)"
                          ? `Both detectors agree${
                              activeDetection.yolo_conf != null && activeDetection.detr_conf != null
                                ? `: YOLO26n ${(activeDetection.yolo_conf * 100).toFixed(0)}%, RT-DETR ${(activeDetection.detr_conf * 100).toFixed(0)}%`
                                : ""
                            }`
                          : `${activeDetection.source} only, not confirmed by the other detector`}
                      </div>
                      <div className="mt-1 font-mono text-stone-700 dark:text-zinc-400">
                        {Math.round(activeDetection.bbox[2] - activeDetection.bbox[0])} ×{" "}
                        {Math.round(activeDetection.bbox[3] - activeDetection.bbox[1])} px at (
                        {Math.round(activeDetection.bbox[0])}, {Math.round(activeDetection.bbox[1])})
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between gap-3 text-[11px] font-mono text-stone-600 dark:text-zinc-400">
            <span>Hover bounding boxes or list items to isolate</span>
            <span className="text-right">
              {imgW && imgH ? `Original: ${imgW} × ${imgH} px` : null}
              {results.timings_ms?.total ? ` · ${(results.timings_ms.total / 1000).toFixed(1)} s` : null}
            </span>
          </div>
        </div>

        {/* Right Column: Telemetry, Recommendations, Anomaly List */}
        <div className="lg:col-span-5 p-6 bg-[#fffdf8] dark:bg-zinc-900/50 flex flex-col space-y-6">
          {/* Resolution check: which path this image took */}
          {route && (
            <div className="p-4 rounded-xl bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800">
              <div className="flex items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-stone-700 dark:text-zinc-400 uppercase tracking-wider">
                  <ScanSearch className="w-3.5 h-3.5" />
                  <span>Resolution Check</span>
                </div>
                <span
                  className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded border ${
                    route.needs_sr
                      ? "bg-amber-50 text-amber-800 border-amber-300 dark:bg-amber-950/30 dark:text-amber-300 dark:border-amber-800/40"
                      : "bg-emerald-50 text-emerald-800 border-emerald-300 dark:bg-emerald-950/30 dark:text-emerald-300 dark:border-emerald-800/40"
                  }`}
                >
                  {route.needs_sr ? "Enhanced before detection" : "Direct, no enhancement"}
                </span>
              </div>
              <p className="text-xs text-stone-800 dark:text-zinc-300 leading-relaxed">
                {route.needs_sr ? `${route.method}. ${route.reason}.` : `${route.reason}.`}
              </p>
              <dl className="mt-2.5 grid grid-cols-3 gap-2 text-[11px] font-mono">
                {(
                  [
                    ["Edge density", route.metrics.edge_density, route.thresholds.edge_density, 3],
                    ["Contrast", route.metrics.contrast, route.thresholds.contrast, 1],
                    ["HF energy", route.metrics.hf_ratio, route.thresholds.hf_ratio, 3],
                  ] as const
                ).map(([name, value, tau, digits]) => (
                  <div key={name} className="rounded-lg bg-stone-100 dark:bg-zinc-800 border border-stone-200 dark:border-zinc-700/60 px-2 py-1.5">
                    <dt className="text-stone-700 dark:text-zinc-400">{name}</dt>
                    <dd className="text-stone-900 dark:text-zinc-100 font-semibold">{value.toFixed(digits)}</dd>
                    <dd className="text-stone-600 dark:text-zinc-400">τ {tau}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {/* Telemetry Metrics */}
          <div>
            <div className="flex items-center gap-1.5 text-xs font-medium text-stone-700 dark:text-zinc-400 uppercase tracking-wider mb-2.5">
              <Cpu className="w-3.5 h-3.5" />
              <span>Model Telemetry</span>
            </div>
            <div className="grid grid-cols-3 gap-2.5">
              <div className="p-3 rounded-xl bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800">
                <span className="text-[10px] font-mono text-stone-700 dark:text-zinc-400 block">YOLO26n</span>
                <span className="text-xl font-semibold font-mono text-stone-900 dark:text-zinc-100">
                  {results.yolo_count}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800">
                <span className="text-[10px] font-mono text-stone-700 dark:text-zinc-400 block">RT-DETR</span>
                <span className="text-xl font-semibold font-mono text-stone-900 dark:text-zinc-100">
                  {results.rtdetr_count}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800">
                <span className="text-[10px] font-mono text-emerald-700 dark:text-emerald-400 block">Fused WBF</span>
                <span className="text-xl font-semibold font-mono text-emerald-700 dark:text-emerald-400">
                  {filteredDetections.length}
                </span>
              </div>
            </div>
          </div>

          {/* Root Cause Recommendations */}
          {results.tips && results.tips.length > 0 && (
            <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/20 border border-amber-300 dark:border-amber-800/40 text-amber-900 dark:text-amber-200">
              <div className="flex items-center gap-2 mb-2 text-amber-800 dark:text-amber-300">
                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                <h3 className="text-xs font-semibold uppercase tracking-wider">
                  Process Diagnostics & Root Cause
                </h3>
              </div>
              <ul className="space-y-1.5 text-xs text-amber-900/90 dark:text-amber-200/90 leading-relaxed list-disc pl-4">
                {results.tips.map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Anomaly Catalog */}
          <div className="flex-1 flex flex-col min-h-[220px]">
            <div className="flex items-center justify-between border-b border-stone-200 dark:border-zinc-800 pb-2 mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-800 dark:text-zinc-300">
                Detected Defect Catalog
              </h3>
              <span className="text-xs font-mono text-stone-700 dark:text-zinc-400">
                {filteredDetections.length} items
              </span>
            </div>

            <AnimatePresence mode="wait">
              {filteredDetections.length === 0 ? (
                <motion.div
                  key="empty-state"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-stone-200 dark:border-zinc-800 text-center my-auto flex flex-col items-center gap-2"
                >
                  <CheckCircle2 className="w-7 h-7 text-emerald-700 dark:text-emerald-400" />
                  <p className="text-xs font-medium text-stone-900 dark:text-zinc-200">Quality Assurance Passed</p>
                  <p className="text-[11px] text-stone-600 dark:text-zinc-400 max-w-xs">
                    No anomalies detected above {(minConf * 100).toFixed(0)}% confidence threshold.
                  </p>
                </motion.div>
              ) : (
                <div ref={listRef} className="relative space-y-2 overflow-y-auto max-h-[340px] pr-1 custom-scrollbar">
                  {filteredDetections.map((det, i) => {
                    const isHovered = hoveredIndex === i;
                    return (
                      <div
                        key={`${det.class}-${i}`}
                        ref={(el) => {
                          itemRefs.current[i] = el;
                        }}
                        tabIndex={0}
                        onMouseEnter={() => setHovered({ index: i, source: "list" })}
                        onMouseLeave={() => setHovered((h) => (h?.index === i ? null : h))}
                        onFocus={() => setHovered({ index: i, source: "list" })}
                        onBlur={() => setHovered((h) => (h?.index === i ? null : h))}
                        className={`p-3 rounded-xl border text-xs transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-sky-500 ${
                          isHovered
                            ? "border-sky-500 bg-sky-50 dark:bg-sky-950/20 text-stone-900 dark:text-zinc-100"
                            : "border-stone-200 dark:border-zinc-800 bg-[#fffdf8] dark:bg-zinc-900/60 hover:border-stone-400 dark:hover:border-zinc-700 text-stone-800 dark:text-zinc-300"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="flex items-center gap-2">
                            <span
                              className={`px-1.5 py-0.5 rounded font-mono font-bold text-[10px] ${
                                isHovered
                                  ? "bg-sky-500 text-black"
                                  : "bg-rose-50 text-rose-700 border border-rose-200 dark:bg-rose-950 dark:text-rose-300 dark:border-rose-800/60"
                              }`}
                            >
                              {det.class}
                            </span>
                            <span className="font-medium text-stone-900 dark:text-zinc-200">{det.full_name}</span>
                          </div>

                          <div className="flex items-center gap-2 font-mono">
                            <span className="text-stone-700 dark:text-zinc-400 text-[11px]">
                              {(det.confidence * 100).toFixed(1)}%
                            </span>
                            {isHovered && <Crosshair className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />}
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-[11px] font-mono text-stone-700 dark:text-zinc-400">
                          <span>Source: {det.source}</span>
                          <span>[{det.bbox.map((n) => Math.round(n)).join(", ")}]</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

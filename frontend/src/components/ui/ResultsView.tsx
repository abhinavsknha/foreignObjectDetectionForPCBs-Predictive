"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  AlertTriangle,
  Cpu,
  SlidersHorizontal,
  CheckCircle2,
  Crosshair,
  Layers,
} from "lucide-react";

interface Detection {
  class: string;
  full_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  source: string;
}

interface ResultsViewProps {
  fileKey: string;
  results: {
    yolo_count: number;
    rtdetr_count: number;
    fused_count: number;
    detections: Detection[];
    tips: string[];
  };
  onReset: () => void;
}

export function ResultsView({ fileKey, results, onReset }: ResultsViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [minConf, setMinConf] = useState(0.45);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  const filteredDetections = results.detections.filter((d) => d.confidence >= minConf);

  const renderCanvas = useCallback(() => {
    if (!canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const imageUrl =
      fileKey.startsWith("blob:") || fileKey.startsWith("data:") || fileKey.startsWith("http")
        ? fileKey
        : `${apiBase}/${fileKey}`;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      const containerWidth = containerRef.current!.clientWidth;
      const scale = containerWidth / img.width;
      const displayWidth = img.width * scale;
      const displayHeight = img.height * scale;

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      // Draw background image
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      // Draw bounding boxes
      filteredDetections.forEach((det, i) => {
        const isHovered = hoveredIndex === i;
        const [x1, y1, x2, y2] = det.bbox;
        const scaledX1 = x1 * scale;
        const scaledY1 = y1 * scale;
        const scaledW = (x2 - x1) * scale;
        const scaledH = (y2 - y1) * scale;

        if (isHovered) {
          // Highlight fill tint
          ctx.fillStyle = "rgba(56, 189, 248, 0.18)";
          ctx.fillRect(scaledX1, scaledY1, scaledW, scaledH);

          // Active border
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 2.5;
          ctx.strokeRect(scaledX1, scaledY1, scaledW, scaledH);

          // Crosshair corner guides
          const cornerLen = Math.min(10, scaledW / 3, scaledH / 3);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          // Top-left
          ctx.moveTo(scaledX1, scaledY1 + cornerLen);
          ctx.lineTo(scaledX1, scaledY1);
          ctx.lineTo(scaledX1 + cornerLen, scaledY1);
          // Top-right
          ctx.moveTo(scaledX1 + scaledW - cornerLen, scaledY1);
          ctx.lineTo(scaledX1 + scaledW, scaledY1);
          ctx.lineTo(scaledX1 + scaledW, scaledY1 + cornerLen);
          // Bottom-left
          ctx.moveTo(scaledX1, scaledY1 + scaledH - cornerLen);
          ctx.lineTo(scaledX1, scaledY1 + scaledH);
          ctx.lineTo(scaledX1 + cornerLen, scaledY1 + scaledH);
          // Bottom-right
          ctx.moveTo(scaledX1 + scaledW - cornerLen, scaledY1 + scaledH);
          ctx.lineTo(scaledX1 + scaledW, scaledY1 + scaledH);
          ctx.lineTo(scaledX1 + scaledW, scaledY1 + scaledH - cornerLen);
          ctx.stroke();
        } else {
          // Standard defect box (rose-500)
          ctx.strokeStyle = "#f43f5e";
          ctx.lineWidth = 1.75;
          ctx.strokeRect(scaledX1, scaledY1, scaledW, scaledH);
        }

        // Draw precision label pill
        const labelText = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "600 11px monospace";
        const textWidth = ctx.measureText(labelText).width;
        const pillWidth = textWidth + 10;
        const pillHeight = 18;
        const pillY = Math.max(0, scaledY1 - pillHeight - 3);

        ctx.fillStyle = isHovered ? "#0284c7" : "rgba(15, 23, 42, 0.9)";
        ctx.fillRect(scaledX1, pillY, pillWidth, pillHeight);

        ctx.strokeStyle = isHovered ? "#38bdf8" : "#f43f5e";
        ctx.lineWidth = 1;
        ctx.strokeRect(scaledX1, pillY, pillWidth, pillHeight);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelText, scaledX1 + 5, pillY + 13);
      });
    };
  }, [fileKey, filteredDetections, hoveredIndex]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Handle canvas click/move to highlight corresponding defect
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageSize) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const scale = canvas.width / imageSize.width;

    const foundIndex = filteredDetections.findIndex((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      return (
        clickX >= x1 * scale &&
        clickX <= x2 * scale &&
        clickY >= y1 * scale &&
        clickY <= y2 * scale
      );
    });

    setHoveredIndex(foundIndex !== -1 ? foundIndex : null);
  };

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

        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-stone-100 dark:bg-zinc-800 hover:bg-stone-200 dark:hover:bg-zinc-700 border border-stone-300 dark:border-zinc-700 text-xs font-medium text-stone-900 dark:text-zinc-200 transition-colors cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Inspect Next Scan</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-stone-200 dark:divide-zinc-800">
        {/* Left Column: Canvas Viewport & Controls */}
        <div className="lg:col-span-7 p-6 bg-[#f7f4ec] dark:bg-zinc-950 flex flex-col justify-between">
          <div className="space-y-4">
            {/* Filter Slider */}
            <div className="flex items-center justify-between gap-4 p-3 rounded-xl bg-[#fffdf8] dark:bg-zinc-900/60 border border-stone-200 dark:border-zinc-800">
              <div className="flex items-center gap-2 text-xs font-medium text-stone-800 dark:text-zinc-300">
                <SlidersHorizontal className="w-4 h-4 text-stone-700 dark:text-zinc-400" />
                <span>Confidence Threshold</span>
              </div>

              <div className="flex items-center gap-3 flex-1 max-w-xs">
                <input
                  type="range"
                  min="20"
                  max="95"
                  value={minConf * 100}
                  onChange={(e) => setMinConf(parseInt(e.target.value) / 100)}
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
              className="w-full relative rounded-xl overflow-hidden border border-stone-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 flex items-center justify-center"
            >
              <canvas
                ref={canvasRef}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={() => setHoveredIndex(null)}
                className="w-full max-h-[560px] object-contain cursor-crosshair block"
              />
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-stone-600 dark:text-zinc-400">
            <span>Hover bounding boxes or list items to isolate</span>
            {imageSize && (
              <span>
                Original: {imageSize.width} × {imageSize.height} px
              </span>
            )}
          </div>
        </div>

        {/* Right Column: Telemetry, Recommendations, Anomaly List */}
        <div className="lg:col-span-5 p-6 bg-[#fffdf8] dark:bg-zinc-900/50 flex flex-col space-y-6">
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
                <div className="space-y-2 overflow-y-auto max-h-[300px] pr-1 custom-scrollbar">
                  {filteredDetections.map((det, i) => {
                    const isHovered = hoveredIndex === i;
                    return (
                      <div
                        key={`${det.class}-${i}`}
                        onMouseEnter={() => setHoveredIndex(i)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className={`p-3 rounded-xl border text-xs transition-colors cursor-pointer ${
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
                          <span>
                            [{det.bbox.map((n) => Math.round(n)).join(", ")}]
                          </span>
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

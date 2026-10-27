"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileUploader } from "@/components/ui/FileUploader";
import { ResultsView } from "@/components/ui/ResultsView";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { PipelineStatus } from "@/components/ui/PipelineStatus";
import type { InferenceResults } from "@/lib/types";
import { Cpu, ShieldCheck } from "lucide-react";

export default function Home() {
  const [inferenceData, setInferenceData] = useState<{ fileKey: string; results: InferenceResults } | null>(null);

  return (
    <div className="min-h-screen bg-[#f7f4ec] dark:bg-zinc-950 text-stone-900 dark:text-zinc-100 flex flex-col font-sans selection:bg-emerald-200 selection:text-emerald-900 dark:selection:bg-emerald-500/30 dark:selection:text-emerald-200">
      {/* Top Engineering App Bar */}
      <header className="border-b border-stone-200 dark:border-zinc-800/80 bg-[#fffdf8] dark:bg-zinc-900/60 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-stone-100 dark:bg-zinc-800 border border-stone-300 dark:border-zinc-700/80 flex items-center justify-center text-emerald-700 dark:text-emerald-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm tracking-tight text-stone-900 dark:text-zinc-100">PCB Optical Inspection</span>
                <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-stone-100 dark:bg-zinc-800 text-stone-800 dark:text-zinc-300 border border-stone-300 dark:border-zinc-700/60">
                  v2.0
                </span>
              </div>
              <p className="text-xs text-stone-700 dark:text-zinc-400 font-mono hidden sm:block">
                Dual-Track Ensemble: YOLO26n + RT-DETR (WBF IoU 0.50)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <PipelineStatus />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col justify-center">
        <AnimatePresence mode="wait">
          {!inferenceData ? (
            <motion.div
              key="uploader-container"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="w-full max-w-3xl mx-auto space-y-6"
            >
              <div className="text-center space-y-2">
                <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-stone-900 dark:text-zinc-100">
                  Automated Defect & Anomaly Inspection
                </h1>
                <p className="text-sm text-stone-700 dark:text-zinc-400 max-w-lg mx-auto">
                  Upload an optical scan or micro-photograph to detect shorts, spurs, mouse bites, and foreign objects with fused bounding boxes.
                </p>
              </div>

              <FileUploader
                onInferenceComplete={(fileKey, results) => setInferenceData({ fileKey, results })}
              />

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                <div className="p-3.5 rounded-xl bg-[#fffdf8] dark:bg-zinc-900/50 border border-stone-200 dark:border-zinc-800/80 text-left">
                  <p className="text-xs font-medium text-stone-800 dark:text-zinc-300">Track 1: YOLO26n</p>
                  <p className="text-xs text-stone-600 dark:text-zinc-400 mt-1">
                    Lightweight convolutional backbone for real-time edge defect localization.
                  </p>
                </div>
                <div className="p-3.5 rounded-xl bg-[#fffdf8] dark:bg-zinc-900/50 border border-stone-200 dark:border-zinc-800/80 text-left">
                  <p className="text-xs font-medium text-stone-800 dark:text-zinc-300">Track 2: RT-DETR</p>
                  <p className="text-xs text-stone-600 dark:text-zinc-400 mt-1">
                    Vision Transformer for global context and high-density defect queries.
                  </p>
                </div>
                <div className="p-3.5 rounded-xl bg-[#fffdf8] dark:bg-zinc-900/50 border border-stone-200 dark:border-zinc-800/80 text-left">
                  <p className="text-xs font-medium text-stone-800 dark:text-zinc-300">Weighted Box Fusion</p>
                  <p className="text-xs text-stone-600 dark:text-zinc-400 mt-1">
                    Coordinate averaging weighted by confidence to suppress false alarms.
                  </p>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="results-container"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              className="w-full"
            >
              <ResultsView
                fileKey={inferenceData.fileKey}
                results={inferenceData.results}
                onReset={() => setInferenceData(null)}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-200 dark:border-zinc-800/60 py-4 text-center text-xs font-mono text-stone-600 dark:text-zinc-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>PCB Surface Defect & Foreign Object Inspection</span>
          <span className="flex items-center gap-1.5 text-stone-700 dark:text-zinc-400">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-700 dark:text-emerald-400" />
            PKU-Market-PCB Benchmark Taxonomy
          </span>
        </div>
      </footer>
    </div>
  );
}

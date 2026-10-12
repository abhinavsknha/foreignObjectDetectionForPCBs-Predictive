"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileUploader } from "@/components/ui/FileUploader";
import { ResultsView } from "@/components/ui/ResultsView";
import { Cpu, ShieldCheck } from "lucide-react";

export default function Home() {
  const [inferenceData, setInferenceData] = useState<{ fileKey: string; results: any } | null>(null);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Engineering App Bar */}
      <header className="border-b border-zinc-800/80 bg-zinc-900/60 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-zinc-800 border border-zinc-700/80 flex items-center justify-center text-emerald-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm tracking-tight text-zinc-100">PCB Optical Inspection</span>
                <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700/60">
                  v2.0
                </span>
              </div>
              <p className="text-xs text-zinc-400 font-mono hidden sm:block">
                Dual-Track Ensemble: YOLO26n + RT-DETR (WBF IoU 0.40)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-xs font-mono text-zinc-300">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              <span>Pipeline Online</span>
            </div>
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
                <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-zinc-100">
                  Automated Defect & Anomaly Inspection
                </h1>
                <p className="text-sm text-zinc-400 max-w-lg mx-auto">
                  Upload an optical scan or micro-photograph to detect shorts, spurs, mouse bites, and foreign objects with fused bounding boxes.
                </p>
              </div>

              <FileUploader
                onInferenceComplete={(fileKey, results) => setInferenceData({ fileKey, results })}
              />

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800/80 text-left">
                  <p className="text-xs font-medium text-zinc-300">Track 1: YOLO26n</p>
                  <p className="text-xs text-zinc-500 mt-1">
                    Lightweight convolutional backbone for real-time edge defect localization.
                  </p>
                </div>
                <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800/80 text-left">
                  <p className="text-xs font-medium text-zinc-300">Track 2: RT-DETR</p>
                  <p className="text-xs text-zinc-500 mt-1">
                    Vision Transformer for global context and high-density defect queries.
                  </p>
                </div>
                <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800/80 text-left">
                  <p className="text-xs font-medium text-zinc-300">Weighted Box Fusion</p>
                  <p className="text-xs text-zinc-500 mt-1">
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
      <footer className="border-t border-zinc-800/60 py-4 text-center text-xs font-mono text-zinc-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>PCB Surface Defect & Foreign Object Inspection</span>
          <span className="flex items-center gap-1.5 text-zinc-400">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            PKU-Market-PCB Benchmark Taxonomy
          </span>
        </div>
      </footer>
    </div>
  );
}

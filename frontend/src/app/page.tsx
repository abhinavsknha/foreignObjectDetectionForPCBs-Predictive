"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileUploader } from "@/components/ui/FileUploader";
import { ResultsView } from "@/components/ui/ResultsView";
import { Sparkles, ScanLine } from "lucide-react";

export default function Home() {
  const [inferenceData, setInferenceData] = useState<{fileKey: string, results: any} | null>(null);

  return (
    <div className="min-h-screen bg-[#f3f3f1] flex flex-col items-center justify-center p-6 font-sans selection:bg-yellow-300 selection:text-black">
      <div className="max-w-6xl w-full space-y-8 my-auto">
        <motion.div 
          className="text-center space-y-3"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
        >
          <motion.div 
            className="inline-block"
            whileHover={{ scale: 1.01, rotate: -0.5 }}
            transition={{ type: "spring", stiffness: 400, damping: 20 }}
          >
            <h1 className="text-4xl md:text-6xl font-black uppercase tracking-tighter text-black border-4 border-black px-8 py-3 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center gap-3">
              <ScanLine className="w-8 h-8 md:w-10 md:h-10 text-black stroke-[3]" />
              PCB Defect Detection
            </h1>
          </motion.div>

          <div>
            <motion.p 
              className="text-sm md:text-base text-black max-w-xl mx-auto font-mono font-bold bg-yellow-300 px-4 py-1.5 border-2 border-black inline-block shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] uppercase tracking-wider"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.15, type: "spring", stiffness: 350, damping: 25 }}
            >
              DUAL-TRACK ENSEMBLE: YOLO26n ➔ RT-DETR ➔ WBF
            </motion.p>
          </div>
        </motion.div>

        <AnimatePresence mode="wait">
          {!inferenceData ? (
            <motion.div
              key="uploader"
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -15 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="w-full flex justify-center"
            >
              <FileUploader onInferenceComplete={(fileKey, results) => setInferenceData({fileKey, results})} />
            </motion.div>
          ) : (
            <motion.div
              key="results"
              initial={{ opacity: 0, scale: 0.96, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 20 }}
              transition={{ type: "spring", stiffness: 280, damping: 24 }}
            >
              <ResultsView 
                fileKey={inferenceData.fileKey} 
                results={inferenceData.results} 
                onReset={() => setInferenceData(null)} 
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

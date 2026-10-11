"use client";
import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, AlertTriangle, Cpu, Merge, SlidersHorizontal, Crosshair, CheckCircle2, Scan } from "lucide-react";

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

  const filteredDetections = results.detections.filter(d => d.confidence >= minConf);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const imageUrl = fileKey.startsWith("blob:") || fileKey.startsWith("data:") || fileKey.startsWith("http")
      ? fileKey
      : `${apiBase}/${fileKey}`;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      const containerWidth = containerRef.current!.clientWidth;
      const scale = containerWidth / img.width;
      const displayWidth = img.width * scale;
      const displayHeight = img.height * scale;

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      filteredDetections.forEach((det, i) => {
        const isHovered = hoveredIndex === i;
        
        const [x1, y1, x2, y2] = det.bbox;
        const scaledX1 = x1 * scale;
        const scaledY1 = y1 * scale;
        const scaledW = (x2 - x1) * scale;
        const scaledH = (y2 - y1) * scale;

        // Highlight logic
        ctx.strokeStyle = isHovered ? "#3b82f6" : "#ef4444"; // Blue if hovered, else Red
        ctx.lineWidth = isHovered ? 6 : 3;
        ctx.strokeRect(scaledX1, scaledY1, scaledW, scaledH);

        // Draw label background
        ctx.fillStyle = isHovered ? "#3b82f6" : "#ef4444";
        const text = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 14px monospace";
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(scaledX1, scaledY1 - 22, textWidth + 8, 22);

        // Draw label text
        ctx.fillStyle = "white";
        ctx.fillText(text, scaledX1 + 4, scaledY1 - 6);
      });
    };
  }, [fileKey, filteredDetections, hoveredIndex]);

  return (
    <motion.div 
      className="w-full max-w-6xl mx-auto bg-white border-4 border-black rounded-none shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] text-black flex flex-col overflow-hidden"
      initial={{ opacity: 0, scale: 0.97, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 280, damping: 25 }}
    >
      {/* Top Banner Header */}
      <div className="flex justify-between items-center border-b-4 border-black p-5 md:p-6 bg-yellow-300">
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            <Merge className="w-7 h-7 md:w-8 md:h-8 stroke-[2.5]" />
          </motion.div>
          <h2 className="text-2xl md:text-3xl font-black uppercase tracking-tighter text-black">
            Ensemble WBF Results
          </h2>
        </div>

        <motion.button 
          onClick={onReset}
          className="flex items-center gap-2 px-5 py-2.5 bg-black text-white font-mono font-bold uppercase tracking-wider border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-sm cursor-pointer"
          whileHover={{ x: -2, y: -2, boxShadow: "6px 6px 0px 0px rgba(0,0,0,1)" }}
          whileTap={{ x: 2, y: 2, boxShadow: "0px 0px 0px 0px rgba(0,0,0,1)" }}
          transition={{ type: "spring", stiffness: 500, damping: 20 }}
        >
          <ArrowLeft className="w-4 h-4 stroke-[3]" /> Inspect Next
        </motion.button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* Left Col: Canvas Image & Slider */}
        <div className="lg:col-span-7 border-r-4 border-black p-6 bg-gray-50 flex flex-col">
           <div className="flex justify-between items-center border-b-3 border-black pb-3 mb-4">
             <h3 className="font-mono font-bold uppercase tracking-wider text-xs md:text-sm text-black flex items-center gap-2">
               <SlidersHorizontal className="w-4 h-4 stroke-[2.5]"/> Confidence Threshold
             </h3>
             <motion.span 
               key={minConf}
               initial={{ scale: 1.25, backgroundColor: "#facc15" }}
               animate={{ scale: 1, backgroundColor: "#000000" }}
               transition={{ type: "spring", stiffness: 450, damping: 25 }}
               className="text-white px-3 py-0.5 text-base md:text-lg font-black font-mono shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"
             >
               {(minConf * 100).toFixed(0)}%
             </motion.span>
           </div>
           
           <input 
             type="range" 
             min="20" max="95" 
             value={minConf * 100} 
             onChange={(e) => setMinConf(parseInt(e.target.value) / 100)}
             className="w-full h-3 bg-gray-300 rounded-none appearance-none cursor-pointer mb-5 border-2 border-black accent-black"
           />
           
           <div ref={containerRef} className="w-full relative overflow-hidden group">
             {/* Subtle animated optical scanning beam */}
             <motion.div 
               className="absolute left-0 right-0 h-0.5 bg-cyan-400 shadow-[0_0_8px_2px_rgba(34,211,238,0.8)] z-10 pointer-events-none"
               animate={{ top: ["0%", "100%", "0%"] }}
               transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
             />

             <canvas ref={canvasRef} className="w-full border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] bg-white mb-3"></canvas>
           </div>
           <p className="font-mono text-[11px] text-gray-700 break-all uppercase font-bold tracking-tight">Source: {fileKey}</p>
        </div>

        {/* Right Col: Data & Tips */}
        <div className="lg:col-span-5 flex flex-col bg-white">
          {/* Comparison Stats */}
          <div className="p-6 border-b-4 border-black bg-white">
            <h3 className="font-black uppercase text-lg mb-3 text-black flex items-center gap-2">
              <Cpu className="w-5 h-5 stroke-[2.5]" /> Dual-Track Pipeline
            </h3>
            <div className="grid grid-cols-3 gap-3 text-center">
              <motion.div 
                className="border-3 border-black p-2.5 bg-gray-50 shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]"
                whileHover={{ y: -2 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
              >
                <p className="font-mono text-[11px] font-bold uppercase text-gray-500">YOLO26n</p>
                <p className="text-2xl font-black mt-0.5">{results.yolo_count}</p>
              </motion.div>

              <motion.div 
                className="border-3 border-black p-2.5 bg-gray-50 shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]"
                whileHover={{ y: -2 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
              >
                <p className="font-mono text-[11px] font-bold uppercase text-gray-500">RT-DETR</p>
                <p className="text-2xl font-black mt-0.5">{results.rtdetr_count}</p>
              </motion.div>

              <motion.div 
                className="border-3 border-black p-2.5 bg-black text-white shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]"
                whileHover={{ y: -2 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
              >
                <p className="font-mono text-[11px] font-bold uppercase text-yellow-300">Filtered WBF</p>
                <motion.p 
                  key={filteredDetections.length}
                  initial={{ scale: 1.35, color: "#facc15" }}
                  animate={{ scale: 1, color: "#ffffff" }}
                  transition={{ type: "spring", stiffness: 450, damping: 20 }}
                  className="text-2xl font-black mt-0.5"
                >
                  {filteredDetections.length}
                </motion.p>
              </motion.div>
            </div>
          </div>

          {/* Actionable Tips */}
          <div className="p-5 border-b-4 border-black bg-red-50">
             <div className="flex items-center gap-2 mb-2.5 text-red-600">
               <motion.div
                 animate={{ scale: [1, 1.18, 1] }}
                 transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
               >
                 <AlertTriangle className="w-5 h-5 stroke-[2.5]" />
               </motion.div>
               <h3 className="font-black uppercase text-sm tracking-wider">Recommended Actions</h3>
             </div>
             <ul className="list-disc pl-5 space-y-1.5">
               {results.tips.map((tip, i) => (
                 <li key={i} className="font-mono text-xs text-black font-bold leading-snug">{tip}</li>
               ))}
             </ul>
          </div>

          {/* Scrollable Anomaly List */}
          <div className="flex-1 flex flex-col p-6 max-h-[420px] overflow-hidden">
            <h3 className="font-black uppercase text-base border-b-3 border-black pb-2 mb-3 text-black flex items-center justify-between">
              <span>Detected Anomalies</span>
              <span className="text-xs font-mono font-bold bg-gray-200 px-2 py-0.5 border border-black">
                {filteredDetections.length} Found
              </span>
            </h3>
            
            <AnimatePresence mode="wait">
              {filteredDetections.length === 0 ? (
                <motion.div 
                  key="empty"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ type: "spring", stiffness: 350, damping: 25 }}
                  className="p-6 bg-green-300 border-3 border-black text-black font-black uppercase text-center my-auto shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] flex flex-col items-center gap-2"
                >
                  <CheckCircle2 className="w-8 h-8 stroke-[2.5]" />
                  <span>QA Passed. Zero defects at {(minConf*100).toFixed(0)}% confidence.</span>
                </motion.div>
              ) : (
                <motion.div 
                  key="list"
                  className="space-y-3 overflow-y-auto pr-3 custom-scrollbar flex-1 pb-2"
                  variants={{
                    hidden: { opacity: 0 },
                    show: { opacity: 1, transition: { staggerChildren: 0.05 } }
                  }}
                  initial="hidden"
                  animate="show"
                >
                  {filteredDetections.map((det, i) => {
                    const isHovered = hoveredIndex === i;
                    return (
                      <motion.div 
                        key={`${det.class}-${i}`}
                        variants={{
                          hidden: { opacity: 0, x: 15 },
                          show: { opacity: 1, x: 0, transition: { type: "spring", stiffness: 380, damping: 24 } }
                        }}
                        whileHover={{ x: 6, transition: { type: "spring", stiffness: 450, damping: 20 } }}
                        whileTap={{ scale: 0.98 }}
                        onMouseEnter={() => setHoveredIndex(i)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className={`border-3 p-3.5 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-black transition-colors cursor-crosshair ${
                          isHovered ? 'bg-blue-50 border-blue-600' : 'bg-white border-black'
                        }`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <p className={`font-black text-lg uppercase leading-tight flex items-center gap-1.5 ${
                              isHovered ? 'text-blue-600' : ''
                            }`}>
                              {isHovered && <Crosshair className="w-4 h-4 text-blue-600 animate-spin" />}
                              {det.class} - {det.full_name}
                            </p>
                            <p className={`font-mono text-xs mt-1 inline-block px-1.5 py-0.5 font-bold ${
                              isHovered ? 'bg-blue-600 text-white' : 'bg-black text-white'
                            }`}>
                              Conf: {(det.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="bg-gray-100 px-2 py-0.5 text-[11px] font-bold font-mono border border-black">
                            {det.source}
                          </div>
                        </div>
                        <div className={`font-mono text-[11px] p-2 border ${
                          isHovered ? 'bg-blue-100 border-blue-400 text-blue-900' : 'bg-gray-50 border-gray-300 text-gray-800'
                        }`}>
                          <span className={`font-bold mr-1.5 ${isHovered ? 'text-blue-700' : 'text-gray-500'}`}>WBF:</span>
                          [{det.bbox.map(n => n.toFixed(0)).join(', ')}]
                        </div>
                      </motion.div>
                    );
                  })}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

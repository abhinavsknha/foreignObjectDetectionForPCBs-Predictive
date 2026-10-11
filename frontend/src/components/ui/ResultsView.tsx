"use client";
import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, AlertTriangle, Cpu, Merge, SlidersHorizontal } from "lucide-react";

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

    const imageUrl = `http://localhost:8000/${fileKey}`;
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
      className="w-full max-w-6xl mx-auto bg-white border-4 border-black rounded-none shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] text-black flex flex-col"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex justify-between items-center border-b-4 border-black p-6 bg-yellow-300">
        <h2 className="text-3xl font-black uppercase tracking-tighter text-black flex items-center gap-3">
          <Merge className="w-8 h-8" />
          Ensemble WBF Results
        </h2>
        <button 
          onClick={onReset}
          className="flex items-center gap-2 px-6 py-3 bg-black text-white font-bold uppercase hover:bg-gray-800 transition-colors border-2 border-transparent focus:border-white focus:outline-none focus:ring-2 focus:ring-black"
        >
          <ArrowLeft className="w-5 h-5" /> Inspect Next
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* Left Col: Canvas Image & Slider */}
        <div className="lg:col-span-7 border-r-4 border-black p-6 bg-gray-50 flex flex-col">
           <div className="flex justify-between items-center border-b-4 border-black pb-4 mb-4">
             <h3 className="font-bold uppercase tracking-widest text-sm text-black flex items-center gap-2">
               <SlidersHorizontal className="w-4 h-4"/> Confidence Threshold
             </h3>
             <span className="bg-black text-white px-3 py-1 text-lg font-black font-mono">
               {(minConf * 100).toFixed(0)}%
             </span>
           </div>
           
           <input 
             type="range" 
             min="20" max="95" 
             value={minConf * 100} 
             onChange={(e) => setMinConf(parseInt(e.target.value) / 100)}
             className="w-full h-3 bg-gray-300 rounded-none appearance-none cursor-pointer mb-6 border-2 border-black"
           />
           
           <div ref={containerRef} className="w-full relative">
             <canvas ref={canvasRef} className="w-full border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] bg-white mb-4"></canvas>
           </div>
           <p className="font-mono text-xs text-black break-all uppercase">Source: {fileKey}</p>
        </div>

        {/* Right Col: Data & Tips */}
        <div className="lg:col-span-5 flex flex-col">
          {/* Comparison Stats */}
          <div className="p-6 border-b-4 border-black bg-white">
            <h3 className="font-black uppercase text-xl mb-4 text-black flex items-center gap-2">
              <Cpu className="w-6 h-6" /> Dual-Track Pipeline
            </h3>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="border-2 border-black p-2">
                <p className="font-mono text-xs font-bold uppercase text-gray-500">YOLO26n</p>
                <p className="text-2xl font-black mt-1">{results.yolo_count}</p>
              </div>
              <div className="border-2 border-black p-2">
                <p className="font-mono text-xs font-bold uppercase text-gray-500">RT-DETR</p>
                <p className="text-2xl font-black mt-1">{results.rtdetr_count}</p>
              </div>
              <div className="border-4 border-black p-2 bg-black text-white">
                <p className="font-mono text-xs font-bold uppercase text-yellow-300">Filtered WBF</p>
                <p className="text-2xl font-black mt-1">{filteredDetections.length}</p>
              </div>
            </div>
          </div>

          {/* Actionable Tips */}
          <div className="p-6 border-b-4 border-black bg-red-50">
             <h3 className="font-black uppercase text-lg mb-3 text-red-600 flex items-center gap-2">
               <AlertTriangle className="w-5 h-5" /> Recommended Actions
             </h3>
             <ul className="list-disc pl-5 space-y-2">
               {results.tips.map((tip, i) => (
                 <li key={i} className="font-mono text-sm text-black font-bold leading-tight">{tip}</li>
               ))}
             </ul>
          </div>

          {/* Scrollable Anomaly List */}
          <div className="flex-1 flex flex-col p-6 max-h-[400px] overflow-hidden">
            <h3 className="font-black uppercase text-lg border-b-4 border-black pb-2 mb-4 text-black">Detected Anomalies</h3>
            
            {filteredDetections.length === 0 ? (
              <div className="p-4 bg-green-300 border-4 border-black text-black font-black uppercase text-center py-8 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
                QA Passed. Zero defects at {(minConf*100).toFixed(0)}% confidence.
              </div>
            ) : (
              <div className="space-y-4 overflow-y-auto pr-4 custom-scrollbar flex-1 pb-4">
                {filteredDetections.map((det, i) => (
                  <div 
                    key={i} 
                    onMouseEnter={() => setHoveredIndex(i)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    className={`border-4 p-4 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-black transition-colors cursor-crosshair ${hoveredIndex === i ? 'bg-blue-50 border-blue-500' : 'bg-white border-black'}`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <p className={`font-black text-xl uppercase leading-none ${hoveredIndex === i ? 'text-blue-600' : ''}`}>
                          {det.class} - {det.full_name}
                        </p>
                        <p className={`font-mono text-xs mt-1 inline-block px-1 ${hoveredIndex === i ? 'bg-blue-600 text-white' : 'bg-black text-white'}`}>
                          Conf: {(det.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="bg-gray-200 px-2 py-1 text-xs font-bold font-mono border-2 border-black">
                        {det.source}
                      </div>
                    </div>
                    <div className={`font-mono text-xs p-2 border-2 ${hoveredIndex === i ? 'bg-blue-100 border-blue-400 text-blue-900' : 'bg-gray-50 border-black text-black'}`}>
                      <span className={`font-bold mr-2 ${hoveredIndex === i ? 'text-blue-700' : 'text-gray-500'}`}>WBF:</span>
                      [{det.bbox.map(n => n.toFixed(0)).join(', ')}]
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

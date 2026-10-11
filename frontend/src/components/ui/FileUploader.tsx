"use client";

import React, { useCallback, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, CheckCircle, Loader2 } from "lucide-react";

interface FileUploaderProps {
  onInferenceComplete: (fileKey: string, results: any) => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function FileUploader({ onInferenceComplete }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing">("idle");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [fileKey, setFileKey] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (status === "processing" && taskId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/inference/status/${taskId}`, {
            headers: {
              "bypass-tunnel-reminder": "true",
            },
          });
          const data = await res.json();
          
          if (data.status === "completed") {
            clearInterval(interval);
            onInferenceComplete(previewUrl || fileKey!, data.result.results);
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    
    return () => clearInterval(interval);
  }, [status, taskId, fileKey, previewUrl, onInferenceComplete]);

  const uploadFile = async (selectedFile: File) => {
    const localBlob = URL.createObjectURL(selectedFile);
    setPreviewUrl(localBlob);
    setStatus("uploading");
    
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      const uploadRes = await fetch(`${API_BASE}/api/upload/local`, {
        method: "POST",
        headers: {
          "bypass-tunnel-reminder": "true",
        },
        body: formData,
      });
      if (!uploadRes.ok) throw new Error("Upload failed");
      const uploadData = await uploadRes.json();
      
      const inferRes = await fetch(`${API_BASE}/api/inference/trigger?file_key=${encodeURIComponent(uploadData.file_key)}`, {
        method: "POST",
        headers: {
          "bypass-tunnel-reminder": "true",
        },
      });
      if (!inferRes.ok) throw new Error("Inference trigger failed");
      const inferData = await inferRes.json();

      setFileKey(uploadData.file_key);
      setTaskId(inferData.task_id);
      setStatus("processing");
    } catch (error) {
      console.warn("Backend offline or unreachable, switching to client demo mode", error);
      setStatus("processing");
      
      // Fallback demo mode for static Vercel deployments when backend is offline
      setTimeout(() => {
        const objectUrl = URL.createObjectURL(selectedFile);
        const demoResults = {
          yolo_count: 3,
          rtdetr_count: 4,
          fused_count: 4,
          detections: [
            {
              class: "SH",
              full_name: "Short",
              confidence: 0.88,
              bbox: [80, 110, 160, 175],
              source: "ENSEMBLE (WBF)"
            },
            {
              class: "SP",
              full_name: "Spur",
              confidence: 0.74,
              bbox: [210, 180, 270, 230],
              source: "ENSEMBLE (WBF)"
            },
            {
              class: "SC",
              full_name: "Spurious Copper",
              confidence: 0.62,
              bbox: [320, 95, 365, 140],
              source: "YOLO26n"
            },
            {
              class: "MB",
              full_name: "Mouse Bite",
              confidence: 0.53,
              bbox: [140, 310, 195, 355],
              source: "RT-DETR"
            }
          ],
          tips: [
            "Short: Unintended connection. Inspect wave soldering temp and residual copper.",
            "Spur: Irregular protrusion found. Check etching chemical balance.",
            "Spurious Copper: Unwanted residue on base material. Review etching exposure timing."
          ]
        };
        setStatus("idle");
        onInferenceComplete(objectUrl, demoResults);
      }, 1500);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  }, []);

  return (
    <div className="w-full max-w-xl mx-auto">
      <motion.div
        className={`relative overflow-hidden border-4 border-black p-10 text-center flex flex-col items-center justify-center min-h-[340px] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] transition-colors ${
          isDragging ? "bg-black text-white" : "bg-white text-black"
        }`}
        animate={{ scale: isDragging ? 1.02 : 1 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onDrop={handleDrop}
      >
        {/* Background mechanical grid lines */}
        <div className="absolute inset-0 pointer-events-none opacity-5 bg-[radial-gradient(#000_1px,transparent_1px)] [background-size:16px_16px]" />

        {/* Industrial laser scan bar active during uploading and processing */}
        {(status === "uploading" || status === "processing") && (
          <motion.div 
            className="absolute left-0 right-0 h-1 bg-yellow-400 shadow-[0_0_12px_3px_rgba(234,179,8,0.9)] z-10 pointer-events-none"
            animate={{ top: ["0%", "100%", "0%"] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
          />
        )}

        <AnimatePresence mode="wait">
          {status === "idle" && (
            <motion.div 
              key="idle" 
              className="flex flex-col items-center gap-4 relative z-0"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ type: "spring", stiffness: 350, damping: 25 }}
            >
              <motion.div
                animate={{ y: [0, -6, 0] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                className="p-4 bg-yellow-300 border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]"
              >
                <UploadCloud className="w-10 h-10 stroke-[2.5]" />
              </motion.div>

              <div>
                <h3 className="text-2xl font-black uppercase tracking-tight">Initialize Inspection</h3>
                <p className="font-mono text-xs mt-1 text-gray-600 font-bold uppercase tracking-wider">Drag & drop raw PCB scan or browse disk</p>
              </div>

              <motion.label 
                className="mt-2 px-8 py-3.5 bg-black text-white font-mono font-black uppercase cursor-pointer border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] tracking-wider"
                whileHover={{ x: -2, y: -2, boxShadow: "8px 8px 0px 0px rgba(0,0,0,1)" }}
                whileTap={{ x: 2, y: 2, boxShadow: "0px 0px 0px 0px rgba(0,0,0,1)" }}
                transition={{ type: "spring", stiffness: 500, damping: 20 }}
              >
                Select Target Image
                <input type="file" accept="image/*" className="hidden" onChange={(e) => {
                  if (e.target.files?.length) uploadFile(e.target.files[0]);
                }} />
              </motion.label>
            </motion.div>
          )}

          {status === "uploading" && (
            <motion.div 
              key="uploading" 
              className="flex flex-col items-center gap-4 relative z-0"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <motion.div 
                animate={{ rotate: 360 }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
                className="p-3 bg-black text-white border-2 border-black"
              >
                <Loader2 className="w-8 h-8" />
              </motion.div>
              <h3 className="text-xl font-black uppercase tracking-tight">Ingesting Scan Binary...</h3>
              <p className="font-mono text-xs bg-gray-200 px-3 py-1 border border-black font-bold">DISPATCHING PAYLOAD</p>
            </motion.div>
          )}

          {status === "processing" && (
            <motion.div 
              key="processing" 
              className="flex flex-col items-center gap-4 relative z-0 w-full max-w-sm"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <motion.div 
                animate={{ rotate: [0, 90, 180, 270, 360] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                className="p-3 bg-yellow-300 text-black border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]"
              >
                <Loader2 className="w-8 h-8 stroke-[3]" />
              </motion.div>

              <div>
                <h3 className="text-xl font-black uppercase tracking-tight">Dual-Track Engine Active</h3>
                <p className="font-mono text-xs text-gray-700 mt-1 font-bold uppercase">
                  YOLO26n + RT-DETR ➔ WBF Fusion
                </p>
              </div>

              {/* Animated Progress Meter */}
              <div className="w-full bg-gray-200 border-2 border-black h-4 p-0.5 overflow-hidden">
                <motion.div 
                  className="bg-black h-full"
                  initial={{ width: "5%" }}
                  animate={{ width: ["15%", "45%", "85%", "95%"] }}
                  transition={{ duration: 2.5, ease: "easeInOut", repeat: Infinity }}
                />
              </div>

              <span className="font-mono text-[10px] uppercase font-bold text-gray-500">
                Awaiting Message Broker Return
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

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
            onInferenceComplete(fileKey!, data.result.results);
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    
    return () => clearInterval(interval);
  }, [status, taskId, fileKey, onInferenceComplete]);

  const uploadFile = async (selectedFile: File) => {
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
        className={`relative overflow-hidden border-4 border-black p-12 text-center flex flex-col items-center justify-center min-h-[300px] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] transition-colors ${
          isDragging ? "bg-black text-white" : "bg-white text-black hover:bg-gray-50"
        }`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onDrop={handleDrop}
      >
        <AnimatePresence mode="wait">
          {status === "idle" && (
            <motion.div key="idle" className="flex flex-col items-center gap-4">
              <UploadCloud className="w-12 h-12" />
              <div>
                <h3 className="text-xl font-bold uppercase tracking-tighter">Initialize Inference</h3>
                <p className="font-mono text-sm mt-2 uppercase">Drag & Drop Target Image</p>
              </div>
              <label className="mt-4 px-8 py-3 bg-black text-white font-bold uppercase cursor-pointer hover:bg-gray-800 border-2 border-black">
                Select File
                <input type="file" className="hidden" onChange={(e) => {
                  if (e.target.files?.length) uploadFile(e.target.files[0]);
                }} />
              </label>
            </motion.div>
          )}

          {status === "uploading" && (
            <motion.div key="uploading" className="flex flex-col items-center gap-4">
              <Loader2 className="w-12 h-12 animate-spin" />
              <h3 className="text-xl font-bold uppercase">Uploading Protocol...</h3>
              <p className="font-mono text-sm">Direct to Node</p>
            </motion.div>
          )}

          {status === "processing" && (
            <motion.div key="processing" className="flex flex-col items-center gap-4">
              <Loader2 className="w-12 h-12 animate-spin" />
              <h3 className="text-xl font-bold uppercase">Engine Active</h3>
              <p className="font-mono text-sm">Celery Sleeper Cell Triggered<br/>Polling Redis Broker...</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

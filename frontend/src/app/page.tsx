"use client";
import { useState } from "react";
import { FileUploader } from "@/components/ui/FileUploader";
import { ResultsView } from "@/components/ui/ResultsView";

export default function Home() {
  const [inferenceData, setInferenceData] = useState<{fileKey: string, results: any} | null>(null);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-6 font-sans">
      <div className="max-w-4xl w-full space-y-12">
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-black uppercase tracking-tighter text-black border-4 border-black inline-block px-8 py-4 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            PCB Defect Detection
          </h1>
          <p className="text-lg text-black max-w-xl mx-auto font-mono font-bold bg-yellow-300 p-2 border-2 border-black">
            DUAL-TRACK ENSEMBLE: YOLO26n ➔ RT-DETR
          </p>
        </div>

        {!inferenceData ? (
          <FileUploader onInferenceComplete={(fileKey, results) => setInferenceData({fileKey, results})} />
        ) : (
          <ResultsView 
            fileKey={inferenceData.fileKey} 
            results={inferenceData.results} 
            onReset={() => setInferenceData(null)} 
          />
        )}
      </div>
    </div>
  );
}

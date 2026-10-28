export interface Detection {
  class: string;
  full_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  source: string;
  yolo_conf?: number | null;
  detr_conf?: number | null;
}

export interface Route {
  needs_sr: boolean;
  gate_flagged?: boolean;
  path: "direct" | "super_resolution" | "reference";
  reason: string;
  method: string | null;
  native: { width: number; height: number };
  metrics: { edge_density: number; contrast: number; hf_ratio: number };
  thresholds: { edge_density: number; contrast: number; hf_ratio: number };
}

export interface InferenceResults {
  yolo_count: number;
  rtdetr_count: number;
  fused_count: number;
  detections: Detection[];
  tips: string[];
  route?: Route;
  image?: { width: number; height: number; green_share?: number };
  timings_ms?: Record<string, number>;
}

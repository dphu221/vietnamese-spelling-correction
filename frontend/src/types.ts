export type CorrectionMode = "conservative" | "balanced" | "aggressive";

export interface Alternative {
  token: string;
  confidence: number;
}

export interface CorrectionItem {
  original: string;
  replacement: string;
  start: number;
  end: number;
  detection_confidence: number;
  correction_confidence: number;
  alternatives: Alternative[];
  error_type: string;
  error_type_label: string;
  explanation_is_inferred: boolean;
}

export interface CorrectionResult {
  original_text: string;
  corrected_text: string;
  corrections: CorrectionItem[];
  processing_ms: number;
  mode: CorrectionMode;
  threshold: number;
  correction_threshold?: number;
  adapter: string;
  model_loaded: boolean;
}

export interface HealthStatus {
  status: "ready" | "unavailable";
  adapter: string;
  source: string;
  model_loaded: boolean;
  device?: string | null;
  detail?: string | null;
}

export interface HistoryEntry {
  id: string;
  createdAt: string;
  result: CorrectionResult;
}

import type { CorrectionMode, CorrectionResult, HealthStatus } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }
  let message = "Không thể kết nối với dịch vụ sửa chính tả.";
  try {
    const payload = await response.json() as { detail?: string | Array<{ msg?: string }> };
    if (typeof payload.detail === "string") {
      message = payload.detail;
    } else if (Array.isArray(payload.detail) && payload.detail[0]?.msg) {
      message = payload.detail[0].msg;
    }
  } catch {
    // Keep the local-friendly fallback message.
  }
  throw new Error(message);
}

export async function getHealth(): Promise<HealthStatus> {
  return parseResponse<HealthStatus>(await fetch(`${API_BASE}/api/health`));
}

export async function correctText(text: string, mode: CorrectionMode): Promise<CorrectionResult> {
  return parseResponse<CorrectionResult>(await fetch(`${API_BASE}/api/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, mode }),
  }));
}

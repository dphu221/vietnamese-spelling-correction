import type { HistoryEntry } from "./types";

export const HISTORY_KEY = "vietnamese-spelling-correction-history-v1";
export const HISTORY_LIMIT = 20;

export function loadHistory(): HistoryEntry[] {
  try {
    const value = localStorage.getItem(HISTORY_KEY);
    if (!value) return [];
    const parsed = JSON.parse(value) as HistoryEntry[];
    return Array.isArray(parsed) ? parsed.slice(0, HISTORY_LIMIT) : [];
  } catch {
    return [];
  }
}

export function storeHistory(entries: HistoryEntry[]): HistoryEntry[] {
  const trimmed = entries.slice(0, HISTORY_LIMIT);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  return trimmed;
}

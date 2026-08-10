import { ChangeEvent, FormEvent, Fragment, useEffect, useRef, useState } from "react";
import { correctText, getHealth } from "./api";
import { HISTORY_KEY, loadHistory, storeHistory } from "./storage";
import type { CorrectionItem, CorrectionMode, CorrectionResult, HealthStatus, HistoryEntry } from "./types";

const MAX_TEXT_LENGTH = 5_000;
const EXAMPLE_TEXT = "Hom nay troi dep, tui ko đi học mà đi chs với bạn. Về nhà mik làm bài nx.";
const MODE_OPTIONS: Array<{ value: CorrectionMode; label: string; description: string }> = [
  { value: "conservative", label: "Ít can thiệp", description: "Tối thiểu 80% · ưu tiên tránh sửa nhầm" },
  { value: "balanced", label: "Tiêu chuẩn", description: "Tối thiểu 50% · phù hợp đa số văn bản" },
  { value: "aggressive", label: "Phát hiện mở rộng", description: "Tối thiểu 30% · chấp nhận nhiều gợi ý hơn" },
];

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function HighlightedText({ text, corrections, corrected }: { text: string; corrections: CorrectionItem[]; corrected: boolean }) {
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  corrections.forEach((change, index) => {
    nodes.push(<Fragment key={`plain-${index}`}>{text.slice(cursor, change.start)}</Fragment>);
    nodes.push(
      <mark className={corrected ? "mark-corrected" : "mark-original"} key={`mark-${index}`}>
        {corrected ? change.replacement : text.slice(change.start, change.end)}
      </mark>,
    );
    cursor = change.end;
  });
  nodes.push(<Fragment key="plain-end">{text.slice(cursor)}</Fragment>);
  return <div className="document-text">{nodes}</div>;
}

function downloadFile(name: string, content: string, type: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [text, setText] = useState("");
  const [mode, setMode] = useState<CorrectionMode>("balanced");
  const [result, setResult] = useState<CorrectionResult | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((reason: Error) => setHealth({
      status: "unavailable",
      adapter: "unavailable",
      source: "local",
      model_loaded: false,
      detail: reason.message,
    }));
  }, []);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    setError("");
    setNotice("");
    if (!text.trim()) {
      setError("Vui lòng nhập văn bản cần kiểm tra.");
      return;
    }
    if (text.length > MAX_TEXT_LENGTH) {
      setError(`Văn bản không được vượt quá ${MAX_TEXT_LENGTH.toLocaleString("vi-VN")} ký tự.`);
      return;
    }
    setBusy(true);
    try {
      const nextResult = await correctText(text, mode);
      setResult(nextResult);
      const entry: HistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        createdAt: new Date().toISOString(),
        result: nextResult,
      };
      setHistory((current) => storeHistory([entry, ...current]));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không thể xử lý văn bản.");
    } finally {
      setBusy(false);
    }
  }

  function clearAll() {
    setText("");
    setResult(null);
    setError("");
    setNotice("");
  }

  async function importText(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!file.name.toLocaleLowerCase("vi").endsWith(".txt")) {
      setError("Chỉ hỗ trợ tệp văn bản định dạng .txt.");
      return;
    }
    const content = await file.text();
    if (content.length > MAX_TEXT_LENGTH) {
      setError(`Tệp vượt quá giới hạn ${MAX_TEXT_LENGTH.toLocaleString("vi-VN")} ký tự.`);
      return;
    }
    setText(content);
    setResult(null);
    setError("");
    setNotice(`Đã tải “${file.name}”.`);
  }

  async function copyResult() {
    if (!result) return;
    await navigator.clipboard.writeText(result.corrected_text);
    setNotice("Đã sao chép văn bản đã sửa.");
  }

  function restoreHistory(entry: HistoryEntry) {
    setText(entry.result.original_text);
    setMode(entry.result.mode);
    setResult(entry.result);
    setError("");
    setNotice("Đã mở lại kết quả trong lịch sử.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-copy">
          <h1>Vietnamese Spelling Correction</h1>
          <p>Phát hiện và gợi ý sửa lỗi chính tả tiếng Việt theo ngữ cảnh.</p>
        </div>
      </header>

      <main>
        <form className="input-card" onSubmit={submit}>
          <div className="card-heading">
            <div>
              <h2>Nhập văn bản</h2>
              <p>Dán một đoạn tiếng Việt hoặc tải tệp .txt từ máy.</p>
            </div>
            <div className="text-actions">
              <input ref={fileInputRef} type="file" accept=".txt,text/plain" onChange={importText} aria-label="Chọn tệp TXT" hidden />
              <button type="button" className="button-quiet" onClick={() => fileInputRef.current?.click()}>Tải tệp .txt</button>
              <button type="button" className="button-quiet" onClick={() => { setText(EXAMPLE_TEXT); setResult(null); setError(""); }}>Văn bản mẫu</button>
              <button type="button" className="button-quiet danger" onClick={clearAll} disabled={!text && !result}>Xóa</button>
            </div>
          </div>

          <label className="textarea-wrap">
            <span className="sr-only">Văn bản cần kiểm tra</span>
            <textarea
              aria-label="Văn bản cần kiểm tra"
              value={text}
              onChange={(event) => { setText(event.target.value); setResult(null); setError(""); }}
              onKeyDown={(event) => {
                if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                  event.preventDefault();
                  void submit();
                }
              }}
              maxLength={MAX_TEXT_LENGTH}
              placeholder="Ví dụ: Hom nay troi dep, tui ko đi học..."
              rows={8}
            />
            <span className="character-count">{text.length.toLocaleString("vi-VN")} / {MAX_TEXT_LENGTH.toLocaleString("vi-VN")}</span>
          </label>

          <div className="form-footer">
            <fieldset className="mode-picker">
              <legend>Mức độ sửa</legend>
              <div className="mode-options">
                {MODE_OPTIONS.map((option) => (
                  <label className={`mode-option ${mode === option.value ? "selected" : ""}`} key={option.value}>
                    <input type="radio" name="mode" value={option.value} checked={mode === option.value} onChange={() => setMode(option.value)} />
                    <span><strong>{option.label}</strong><small>{option.description}</small></span>
                  </label>
                ))}
              </div>
            </fieldset>
            <button className="button-primary" type="submit" aria-label="Kiểm tra chính tả" disabled={busy || !text.trim()}>
              {busy ? "Đang phân tích…" : "Kiểm tra chính tả"}
              <kbd>Ctrl ↵</kbd>
            </button>
          </div>
          {error && <div className="message error-message" role="alert">{error}</div>}
          {notice && <div className="message notice-message" role="status">{notice}</div>}
          {health?.detail && health.adapter === "demo" && <p className="demo-note">{health.detail}</p>}
          {health?.detail && health.status === "unavailable" && <div className="message error-message" role="alert">{health.detail}</div>}
        </form>

        {result ? (
          <section className="results" aria-live="polite">
            <div className="section-heading">
              <div>
                <h2>Kết quả kiểm tra</h2>
                <p>{result.corrections.length ? `Đã tìm thấy ${result.corrections.length} vị trí có thể cần sửa.` : "Không phát hiện lỗi cần sửa ở mức đã chọn."}</p>
              </div>
              <div className="result-meta">
                <span>{result.processing_ms.toLocaleString("vi-VN")} ms</span>
                <span>Phát hiện {formatPercent(result.threshold)}</span>
                <span>Gợi ý {formatPercent(result.correction_threshold ?? result.threshold)}</span>
              </div>
            </div>

            <div className="comparison-grid">
              <article className="document-panel">
                <header><h3>Văn bản gốc</h3><span>{result.original_text.length} ký tự</span></header>
                <HighlightedText text={result.original_text} corrections={result.corrections} corrected={false} />
              </article>
              <article className="document-panel corrected-panel">
                <header>
                  <h3>Văn bản đã sửa</h3>
                  <div className="inline-actions">
                    <button type="button" onClick={() => void copyResult()}>Sao chép</button>
                    <button type="button" onClick={() => downloadFile("van-ban-da-sua.txt", result.corrected_text, "text/plain;charset=utf-8")}>Tải TXT</button>
                    <button type="button" onClick={() => downloadFile("bao-cao-sua-loi.json", JSON.stringify(result, null, 2), "application/json;charset=utf-8")}>Tải JSON</button>
                  </div>
                </header>
                <HighlightedText text={result.original_text} corrections={result.corrections} corrected />
              </article>
            </div>

            {result.corrections.length > 0 && (
              <div className="changes-card">
                <div className="changes-heading">
                  <div><h3>Chi tiết thay đổi</h3><p>Loại lỗi là giải thích suy đoán từ cặp từ trước và sau, không phải nhãn do mô hình dự đoán.</p></div>
                  <span className="count-badge">{result.corrections.length} thay đổi</span>
                </div>
                <div className="change-list">
                  {result.corrections.map((change, index) => (
                    <article className="change-row" key={`${change.start}-${index}`}>
                      <span className="change-index">{String(index + 1).padStart(2, "0")}</span>
                      <div className="word-change"><del>{change.original}</del><span aria-hidden="true">→</span><ins>{change.replacement}</ins></div>
                      <div className="error-kind"><strong>{change.error_type_label}</strong><small>Giải thích suy đoán</small></div>
                      <div className="confidence">
                        <span><strong>{formatPercent(change.detection_confidence)}</strong> phát hiện</span>
                        <span className="confidence-track"><i style={{ width: formatPercent(change.detection_confidence) }} /></span>
                      </div>
                      <div className="alternatives">
                        <small>Gợi ý gần nhất</small>
                        <div>{change.alternatives.map((item) => <span key={item.token}>{item.token} <em>{formatPercent(item.confidence)}</em></span>)}</div>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            )}
          </section>
        ) : (
          <section className="empty-state">
            <span aria-hidden="true">Aa</span>
            <div><h2>Kết quả sẽ xuất hiện tại đây</h2><p>Nhập văn bản phía trên và chọn “Kiểm tra chính tả”.</p></div>
          </section>
        )}

        <section className="history-card">
          <div className="section-heading compact">
            <div><h2>Lịch sử trên thiết bị</h2><p>Lưu tối đa 20 lần kiểm tra trong trình duyệt này.</p></div>
            {history.length > 0 && <button className="button-quiet danger" type="button" onClick={() => { localStorage.removeItem(HISTORY_KEY); setHistory([]); }}>Xóa lịch sử</button>}
          </div>
          {history.length === 0 ? <p className="history-empty">Chưa có kết quả nào được lưu.</p> : (
            <div className="history-list">
              {history.map((entry) => (
                <button type="button" className="history-row" key={entry.id} onClick={() => restoreHistory(entry)}>
                  <span>{entry.result.original_text.slice(0, 90)}{entry.result.original_text.length > 90 ? "…" : ""}</span>
                  <small>{new Date(entry.createdAt).toLocaleString("vi-VN")} · {entry.result.corrections.length} thay đổi</small>
                </button>
              ))}
            </div>
          )}
        </section>
      </main>

      <footer><span>Chạy cục bộ · Dữ liệu văn bản không được lưu trên máy chủ</span><span>Mô hình giới hạn sửa một token thành một token</span></footer>
    </div>
  );
}

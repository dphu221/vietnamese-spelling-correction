import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import App from "./App";
import type { CorrectionResult } from "./types";

const result: CorrectionResult = {
  original_text: "tui ko đi học",
  corrected_text: "tôi không đi học",
  corrections: [
    {
      original: "tui",
      replacement: "tôi",
      start: 0,
      end: 3,
      detection_confidence: 0.96,
      correction_confidence: 0.93,
      alternatives: [{ token: "tôi", confidence: 0.93 }],
      error_type: "teencode",
      error_type_label: "Teencode / viết tắt",
      explanation_is_inferred: true,
    },
    {
      original: "ko",
      replacement: "không",
      start: 4,
      end: 6,
      detection_confidence: 0.99,
      correction_confidence: 0.97,
      alternatives: [{ token: "không", confidence: 0.97 }],
      error_type: "teencode",
      error_type_label: "Teencode / viết tắt",
      explanation_is_inferred: true,
    },
  ],
  processing_ms: 3.5,
  mode: "balanced",
  threshold: 0.5,
  correction_threshold: 0.5,
  adapter: "demo",
  model_loaded: false,
};

function mockApi() {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "ready", adapter: "demo", source: "demo", model_loaded: false, detail: "Chế độ minh họa" }) })
    .mockResolvedValueOnce({ ok: true, json: async () => result });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

test("corrects text, copies output, clears, and restores history", async () => {
  const fetchMock = mockApi();
  const user = userEvent.setup();
  const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);
  render(<App />);
  await screen.findByText("Chế độ minh họa");
  expect(screen.queryByText("ĐỒ ÁN XỬ LÝ NGÔN NGỮ TỰ NHIÊN")).not.toBeInTheDocument();
  expect(screen.queryByText("Mô hình đã sẵn sàng")).not.toBeInTheDocument();
  expect(screen.queryByText("01")).not.toBeInTheDocument();
  expect(screen.getByText("Ít can thiệp")).toBeInTheDocument();
  expect(screen.getByText("Tiêu chuẩn")).toBeInTheDocument();
  expect(screen.getByText("Phát hiện mở rộng")).toBeInTheDocument();
  expect(screen.getByText("Tối thiểu 50% · phù hợp đa số văn bản")).toBeInTheDocument();
  const textarea = screen.getByRole("textbox", { name: "Văn bản cần kiểm tra" });
  await user.type(textarea, "tui ko đi học");
  await user.click(screen.getByRole("button", { name: "Kiểm tra chính tả" }));
  expect(await screen.findByText("Chi tiết thay đổi")).toBeInTheDocument();
  expect(fetchMock).toHaveBeenLastCalledWith("/api/correct", expect.objectContaining({ method: "POST" }));
  await user.click(screen.getByRole("button", { name: "Sao chép" }));
  expect(writeText).toHaveBeenCalledWith("tôi không đi học");
  await user.click(screen.getByRole("button", { name: "Xóa" }));
  expect(textarea).toHaveValue("");
  await user.click(screen.getByRole("button", { name: /tui ko đi học/ }));
  expect(textarea).toHaveValue("tui ko đi học");
});

test("imports only short TXT files and exposes downloads", async () => {
  mockApi();
  const user = userEvent.setup();
  const createObjectURL = vi.fn(() => "blob:test");
  vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL: vi.fn() });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  render(<App />);
  await screen.findByText("Chế độ minh họa");
  const fileInput = screen.getByLabelText("Chọn tệp TXT");
  const file = new File(["tui ko đi học"], "input.txt", { type: "text/plain" });
  fireEvent.change(fileInput, { target: { files: [file] } });
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Văn bản cần kiểm tra" })).toHaveValue("tui ko đi học"));
  await user.click(screen.getByRole("button", { name: "Kiểm tra chính tả" }));
  await screen.findByText("Chi tiết thay đổi");
  await user.click(screen.getByRole("button", { name: "Tải TXT" }));
  expect(createObjectURL).toHaveBeenCalled();
});

import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";

Object.defineProperty(window, "scrollTo", { configurable: true, value: vi.fn() });

afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

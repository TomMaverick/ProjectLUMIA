import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

function createMockCanvasContext(): CanvasRenderingContext2D {
  return {
    fillStyle: "#000000",
    strokeStyle: "#ffffff",
    lineWidth: 1,
    font: "12px monospace",
    imageSmoothingEnabled: false,
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    fillText: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    scale: vi.fn(),
    setLineDash: vi.fn(),
    drawImage: vi.fn(),
    createImageData: (width: number, height: number): ImageData => ({
      width,
      height,
      colorSpace: "srgb",
      data: new Uint8ClampedArray(width * height * 4),
    }),
    putImageData: vi.fn(),
  } as unknown as CanvasRenderingContext2D;
}

Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  configurable: true,
  value: vi.fn(() => createMockCanvasContext()),
});

import React, { useEffect, useRef } from "react";
import type { ColorMapMode, StretchMode } from "../../types/api";
import { mapIntensityToRgb, normalizeCutoutPixels } from "../../utils/astronomyFormat";

export interface SingleChannelStampProps {
  readonly channelLabel: string;
  readonly channelSub: string;
  readonly pixels: readonly (readonly number[])[] | null | undefined;
  readonly cutoutSize: number;
  readonly targetX: number;
  readonly targetY: number;
  readonly colorMap: ColorMapMode;
  readonly stretchMode: StretchMode;
  readonly showReticle: boolean;
  readonly canvasSizePx?: number;
}

export const SingleChannelStamp: React.FC<SingleChannelStampProps> = ({
  channelLabel,
  channelSub,
  pixels,
  cutoutSize,
  targetX,
  targetY,
  colorMap,
  stretchMode,
  showReticle,
  canvasSizePx = 200,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#090a0f";
    ctx.fillRect(0, 0, canvasSizePx, canvasSizePx);

    if (!pixels || pixels.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("CHANNEL UNAVAILABLE", canvasSizePx / 2, canvasSizePx / 2);
      return;
    }

    const normalized = normalizeCutoutPixels(pixels, stretchMode);
    const gridHeight = normalized.length;
    const gridWidth = normalized[0]?.length ?? cutoutSize;

    if (gridHeight === 0 || gridWidth === 0) {
      return;
    }

    const cellW = canvasSizePx / gridWidth;
    const cellH = canvasSizePx / gridHeight;

    for (let y = 0; y < gridHeight; y += 1) {
      const row = normalized[y];
      if (!row) {
        continue;
      }
      for (let x = 0; x < gridWidth; x += 1) {
        const val = row[x] ?? 0;
        ctx.fillStyle = mapIntensityToRgb(val, colorMap);
        ctx.fillRect(
          Math.floor(x * cellW),
          Math.floor(y * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH),
        );
      }
    }

    if (showReticle && Number.isFinite(targetX) && Number.isFinite(targetY)) {
      const targetCanvasX = (targetX + 0.5) * cellW;
      const targetCanvasY = (targetY + 0.5) * cellH;
      const innerGap = 6;
      const armLength = 14;

      ctx.save();
      ctx.strokeStyle = "#c084fc";
      ctx.lineWidth = 1.4;

      ctx.beginPath();
      ctx.moveTo(targetCanvasX - armLength, targetCanvasY);
      ctx.lineTo(targetCanvasX - innerGap, targetCanvasY);
      ctx.moveTo(targetCanvasX + innerGap, targetCanvasY);
      ctx.lineTo(targetCanvasX + armLength, targetCanvasY);
      ctx.moveTo(targetCanvasX, targetCanvasY - armLength);
      ctx.lineTo(targetCanvasX, targetCanvasY - innerGap);
      ctx.moveTo(targetCanvasX, targetCanvasY + innerGap);
      ctx.lineTo(targetCanvasX, targetCanvasY + armLength);
      ctx.stroke();

      ctx.strokeStyle = "rgba(168, 85, 247, 0.75)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.arc(targetCanvasX, targetCanvasY, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, [pixels, cutoutSize, targetX, targetY, colorMap, stretchMode, showReticle, canvasSizePx]);

  return (
    <div className="flex flex-col items-center rounded border border-[#26293d] bg-[#090a0f] p-2 shadow-inner">
      <div className="flex w-full items-baseline justify-between border-b border-[#181b28] pb-1 mb-1.5 px-0.5">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-200">
          {channelLabel}
        </span>
        <span className="text-[9px] font-mono uppercase text-slate-400">
          {channelSub}
        </span>
      </div>

      <div className="relative overflow-hidden rounded border border-[#3b185f]">
        <canvas
          ref={canvasRef}
          width={canvasSizePx}
          height={canvasSizePx}
          className="block"
          style={{ imageRendering: "pixelated", width: `${canvasSizePx}px`, height: `${canvasSizePx}px` }}
        />
      </div>
    </div>
  );
};

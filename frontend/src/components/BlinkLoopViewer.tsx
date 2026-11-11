import React, { useEffect, useRef, useState } from "react";
import type {
  CandidateCutoutSequenceResponse,
  ColorMapMode,
  StretchMode,
} from "../types/api";
import {
  mapIntensityToRgb,
  normalizeCutoutPixels,
} from "../utils/astronomyFormat";

export interface BlinkLoopViewerProps {
  readonly cutoutSequence: CandidateCutoutSequenceResponse | null;
  readonly isLoading: boolean;
  readonly activeEpochIndex: number;
  readonly onEpochChange: (nextIndex: number) => void;
}

const CANVAS_SIZE_PX = 320;
const CADENCE_OPTIONS_HZ: readonly number[] = [1, 2, 4, 8];

export function getSurveyBadgeDetails(surveyId?: string): {
  readonly label: string;
  readonly badgeClass: string;
  readonly shortName: string;
} {
  const sid = (surveyId ?? "ZTF").toUpperCase();
  if (sid.includes("PAN") || sid.includes("PS1")) {
    return {
      label: "Pan-STARRS",
      badgeClass: "border-cyan-500 bg-cyan-950/70 text-cyan-200",
      shortName: "PS1",
    };
  }
  if (sid.includes("RUBIN") || sid.includes("LSST")) {
    return {
      label: "Rubin / LSST",
      badgeClass: "border-emerald-500 bg-emerald-950/70 text-emerald-200",
      shortName: "LSST",
    };
  }
  if (sid.includes("JWST")) {
    return {
      label: "JWST (NIRCam)",
      badgeClass: "border-indigo-400 bg-indigo-950/70 text-indigo-200",
      shortName: "JWST",
    };
  }
  if (sid.includes("ROMAN") || sid.includes("EUCLID") || sid.includes("SPACE")) {
    return {
      label: surveyId ?? "Spacecraft",
      badgeClass: "border-pink-500 bg-pink-950/70 text-pink-200",
      shortName: "SPACE",
    };
  }
  return {
    label: surveyId ?? "ZTF",
    badgeClass: "border-amber-500 bg-amber-950/70 text-amber-200",
    shortName: "ZTF",
  };
}

export const BlinkLoopViewer: React.FC<BlinkLoopViewerProps> = ({
  cutoutSequence,
  isLoading,
  activeEpochIndex,
  onEpochChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [cadenceHz, setCadenceHz] = useState<number>(2);
  const [stretchMode, setStretchMode] = useState<StretchMode>("zscale");
  const [colorMap, setColorMap] = useState<ColorMapMode>("violet");
  const [showReticle, setShowReticle] = useState<boolean>(true);
  const [showMotionTrack, setShowMotionTrack] = useState<boolean>(true);
  const [normalizeFov, setNormalizeFov] = useState<boolean>(true);
  const [showScaleBar, setShowScaleBar] = useState<boolean>(true);

  const epochCount = cutoutSequence?.epochs.length ?? 0;
  const safeEpochIndex =
    epochCount > 0 ? Math.min(activeEpochIndex, epochCount - 1) : 0;
  const currentEpoch = cutoutSequence?.epochs[safeEpochIndex] ?? null;

  // Survey classification & multi-survey detection
  const distinctSurveys = Array.from(
    new Set(
      cutoutSequence?.epochs
        .map((ep) => ep.survey_id ?? cutoutSequence.survey_id ?? "ZTF")
        .filter(Boolean),
    ),
  );
  const isMultiSurvey = distinctSurveys.length > 1;

  const currentSurveyId =
    currentEpoch?.survey_id ?? cutoutSequence?.survey_id ?? "ZTF";
  const currentFilterBand = currentEpoch?.filter_band ?? "r";
  const activeSurveyBadge = getSurveyBadgeDetails(currentSurveyId);
  const currentScaleArcsec = currentEpoch?.pixel_scale_arcsec ?? 1.012;

  // Maximum physical FOV [arcseconds] across sequence for normalized coordinate alignment
  const maxFovArcsec = Math.max(
    8.0,
    ...(cutoutSequence?.epochs.map((ep) => {
      const scale = ep.pixel_scale_arcsec ?? 1.012;
      const dim = ep.pixels[0]?.length ?? cutoutSequence.cutout_size;
      return ep.fov_arcsec ?? dim * scale;
    }) ?? [32.0]),
  );

  // Automatic blink-loop timer
  useEffect(() => {
    if (!isPlaying || epochCount <= 1) {
      return undefined;
    }
    const intervalMs = Math.max(80, Math.round(1000 / cadenceHz));
    const timerId = globalThis.setInterval(() => {
      onEpochChange((safeEpochIndex + 1) % epochCount);
    }, intervalMs);

    return () => {
      globalThis.clearInterval(timerId);
    };
  }, [isPlaying, cadenceHz, epochCount, safeEpochIndex, onEpochChange]);

  // Render active astronomical FITS cutout & overlays onto HTML5 2D Canvas
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
    ctx.fillRect(0, 0, CANVAS_SIZE_PX, CANVAS_SIZE_PX);

    if (!cutoutSequence || !currentEpoch) {
      return;
    }

    const normalized = normalizeCutoutPixels(currentEpoch.pixels, stretchMode);
    const gridHeight = normalized.length;
    const gridWidth = normalized[0]?.length ?? cutoutSequence.cutout_size;
    if (gridHeight === 0 || gridWidth === 0) {
      return;
    }

    const scaleArcsec = currentEpoch.pixel_scale_arcsec ?? 1.012;
    let cellW = CANVAS_SIZE_PX / gridWidth;
    let cellH = CANVAS_SIZE_PX / gridHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (normalizeFov && maxFovArcsec > 0) {
      cellW = (scaleArcsec / maxFovArcsec) * CANVAS_SIZE_PX;
      cellH = (scaleArcsec / maxFovArcsec) * CANVAS_SIZE_PX;
      const renderedW = gridWidth * cellW;
      const renderedH = gridHeight * cellH;
      offsetX = Math.max(0, (CANVAS_SIZE_PX - renderedW) / 2);
      offsetY = Math.max(0, (CANVAS_SIZE_PX - renderedH) / 2);
    }

    // Render astronomical raster pixels
    for (let y = 0; y < gridHeight; y += 1) {
      const row = normalized[y];
      if (!row) {
        continue;
      }
      for (let x = 0; x < gridWidth; x += 1) {
        const val = row[x] ?? 0;
        ctx.fillStyle = mapIntensityToRgb(val, colorMap);
        ctx.fillRect(
          Math.floor(offsetX + x * cellW),
          Math.floor(offsetY + y * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH),
        );
      }
    }

    // Instrument aperture boundary when scaled in normalized FOV mode
    if (normalizeFov && (offsetX > 0 || offsetY > 0)) {
      ctx.save();
      ctx.strokeStyle = "rgba(168, 85, 247, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(
        Math.floor(offsetX),
        Math.floor(offsetY),
        Math.ceil(gridWidth * cellW),
        Math.ceil(gridHeight * cellH),
      );
      ctx.restore();
    }

    // Inter-epoch motion trajectory vector overlay
    if (showMotionTrack && cutoutSequence.epochs.length > 1) {
      ctx.save();
      ctx.strokeStyle = "rgba(168, 85, 247, 0.55)";
      ctx.lineWidth = 1.25;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();

      cutoutSequence.epochs.forEach((ep, idx) => {
        let epCellW = cellW;
        let epCellH = cellH;
        let epOffX = offsetX;
        let epOffY = offsetY;

        if (normalizeFov && maxFovArcsec > 0) {
          const epScale = ep.pixel_scale_arcsec ?? 1.012;
          epCellW = (epScale / maxFovArcsec) * CANVAS_SIZE_PX;
          epCellH = (epScale / maxFovArcsec) * CANVAS_SIZE_PX;
          const rW = (ep.pixels[0]?.length ?? cutoutSequence.cutout_size) * epCellW;
          const rH = (ep.pixels.length || cutoutSequence.cutout_size) * epCellH;
          epOffX = Math.max(0, (CANVAS_SIZE_PX - rW) / 2);
          epOffY = Math.max(0, (CANVAS_SIZE_PX - rH) / 2);
        }

        const px = epOffX + (ep.target_x_px + 0.5) * epCellW;
        const py = epOffY + (ep.target_y_px + 0.5) * epCellH;
        if (idx === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      });
      ctx.stroke();
      ctx.setLineDash([]);

      for (const ep of cutoutSequence.epochs) {
        if (ep.epoch_index === currentEpoch.epoch_index) {
          continue;
        }
        let epCellW = cellW;
        let epCellH = cellH;
        let epOffX = offsetX;
        let epOffY = offsetY;

        if (normalizeFov && maxFovArcsec > 0) {
          const epScale = ep.pixel_scale_arcsec ?? 1.012;
          epCellW = (epScale / maxFovArcsec) * CANVAS_SIZE_PX;
          epCellH = (epScale / maxFovArcsec) * CANVAS_SIZE_PX;
          const rW = (ep.pixels[0]?.length ?? cutoutSequence.cutout_size) * epCellW;
          const rH = (ep.pixels.length || cutoutSequence.cutout_size) * epCellH;
          epOffX = Math.max(0, (CANVAS_SIZE_PX - rW) / 2);
          epOffY = Math.max(0, (CANVAS_SIZE_PX - rH) / 2);
        }

        const px = epOffX + (ep.target_x_px + 0.5) * epCellW;
        const py = epOffY + (ep.target_y_px + 0.5) * epCellH;
        ctx.fillStyle = "rgba(192, 132, 252, 0.65)";
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Astronomical aperture & crosshair reticle centered on moving target
    if (showReticle) {
      const targetCanvasX = offsetX + (currentEpoch.target_x_px + 0.5) * cellW;
      const targetCanvasY = offsetY + (currentEpoch.target_y_px + 0.5) * cellH;
      const innerGap = 8;
      const armLength = 18;

      ctx.save();
      ctx.strokeStyle = "#c084fc";
      ctx.lineWidth = 1.6;

      // Crosshair arms with unobstructed central core
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

      // Photometric aperture circle
      ctx.strokeStyle = "rgba(168, 85, 247, 0.85)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(targetCanvasX, targetCanvasY, 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Physical astrometric scale bar (arcseconds)
    if (showScaleBar) {
      const visibleArcsec = normalizeFov
        ? maxFovArcsec
        : gridWidth * scaleArcsec;
      const candidates = [1, 2, 5, 10, 15, 20, 30];
      const targetBar = visibleArcsec * 0.25;
      const barArcsec = candidates.reduce((prev, curr) =>
        Math.abs(curr - targetBar) < Math.abs(prev - targetBar) ? curr : prev,
      );
      const barCanvasPx = (barArcsec / visibleArcsec) * CANVAS_SIZE_PX;

      const barX = 12;
      const barY = CANVAS_SIZE_PX - 12;

      ctx.save();
      ctx.fillStyle = "rgba(9, 10, 15, 0.75)";
      ctx.fillRect(barX - 4, barY - 16, barCanvasPx + 8, 20);
      ctx.strokeStyle = "rgba(90, 40, 140, 0.7)";
      ctx.strokeRect(barX - 4, barY - 16, barCanvasPx + 8, 20);

      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      ctx.moveTo(barX, barY);
      ctx.lineTo(barX + barCanvasPx, barY);
      ctx.moveTo(barX, barY - 4);
      ctx.lineTo(barX, barY + 4);
      ctx.moveTo(barX + barCanvasPx, barY - 4);
      ctx.lineTo(barX + barCanvasPx, barY + 4);
      ctx.stroke();

      ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(
        `${barArcsec}"`,
        barX + Math.max(0, barCanvasPx / 2 - 5),
        barY - 5,
      );
      ctx.restore();
    }
  }, [
    cutoutSequence,
    currentEpoch,
    stretchMode,
    colorMap,
    showReticle,
    showMotionTrack,
    normalizeFov,
    showScaleBar,
    maxFovArcsec,
  ]);

  const handleStepPrev = (): void => {
    if (epochCount === 0) {
      return;
    }
    onEpochChange((safeEpochIndex - 1 + epochCount) % epochCount);
  };

  const handleStepNext = (): void => {
    if (epochCount === 0) {
      return;
    }
    onEpochChange((safeEpochIndex + 1) % epochCount);
  };

  const currentFovDisplay = currentEpoch
    ? (
        currentEpoch.fov_arcsec ??
        (currentEpoch.pixels[0]?.length ?? cutoutSequence?.cutout_size ?? 31) *
          currentScaleArcsec
      ).toFixed(1)
    : "--";

  return (
    <section
      aria-labelledby="blink-viewer-heading"
      className="flex flex-col rounded-lg border border-[#3b185f] bg-[#11131c] p-4 shadow-lg shadow-black/40"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#26293d] pb-3">
        <div className="flex flex-col gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2
              id="blink-viewer-heading"
              className="text-sm font-semibold uppercase tracking-wider text-slate-100"
            >
              Blink-Loop Cutout Inspector
            </h2>
            {isMultiSurvey && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-purple-500/70 bg-purple-950/80 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-purple-200">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Multi-Survey Fusion
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400">
            {cutoutSequence
              ? `Target ${cutoutSequence.candidate_id} | Field ${cutoutSequence.field_id} (${cutoutSequence.cutout_size}x${cutoutSequence.cutout_size} px)`
              : "Select a candidate to inspect multi-epoch cutouts"}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Active Survey & Band Badge */}
          {currentEpoch && (
            <div className="flex items-center gap-1.5">
              <span
                className={`rounded border px-2 py-0.5 text-xs font-semibold ${activeSurveyBadge.badgeClass}`}
              >
                {activeSurveyBadge.label}
              </span>
              <span className="rounded border border-[#581c87] bg-[#22123b] px-1.5 py-0.5 font-mono text-xs font-medium text-purple-200">
                band: {currentFilterBand}
              </span>
            </div>
          )}

          <div className="rounded border border-[#581c87] bg-[#3b185f]/50 px-2.5 py-1 text-xs font-semibold text-purple-200 tabular-nums">
            {epochCount > 0
              ? `Epoch ${safeEpochIndex + 1} / ${epochCount}`
              : "Epoch 0 / 0"}
          </div>
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className="mt-4 flex flex-col items-center">
        <div className="relative overflow-hidden rounded-md border-2 border-[#3b185f] bg-[#090a0f] shadow-inner">
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE_PX}
            height={CANVAS_SIZE_PX}
            aria-label="Sequential astronomical cutout blink-loop canvas"
            className="block h-[280px] w-[280px] sm:h-[320px] sm:w-[320px]"
          />

          {/* Overlaid FOV & Scale Badge */}
          {currentEpoch && (
            <div className="pointer-events-none absolute right-2 top-2 rounded border border-[#3b185f]/90 bg-[#090a0f]/80 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur-sm">
              <span>{currentScaleArcsec.toFixed(3)}&quot;/px</span>
              <span className="mx-1.5 text-slate-500">|</span>
              <span>FOV: {currentFovDisplay}&quot;</span>
            </div>
          )}

          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-[#090a0f]/80 text-xs font-medium text-purple-200">
              Loading epoch cutouts...
            </div>
          )}
        </div>

        {/* Active Epoch Telemetry Bar */}
        <div className="mt-3 grid w-full grid-cols-2 gap-2 rounded-md border border-[#26293d] bg-[#090a0f] px-3 py-2 text-xs tabular-nums sm:grid-cols-4">
          <div>
            <span className="block text-[10px] uppercase text-slate-400">
              Survey / Filter
            </span>
            <span className="font-mono text-slate-200">
              {currentEpoch
                ? `${activeSurveyBadge.shortName} (${currentFilterBand})`
                : "--"}
            </span>
          </div>

          <div>
            <span className="block text-[10px] uppercase text-slate-400">
              BJD (TDB)
            </span>
            <span className="font-mono text-slate-200">
              {currentEpoch ? currentEpoch.bjd_tdb.toFixed(5) : "--"}
            </span>
          </div>

          <div>
            <span className="block text-[10px] uppercase text-slate-400">
              Centroid (X, Y)
            </span>
            <span className="font-mono text-slate-200">
              {currentEpoch
                ? `${currentEpoch.target_x_px.toFixed(1)}, ${currentEpoch.target_y_px.toFixed(1)} px`
                : "--"}
            </span>
          </div>

          <div>
            <span className="block text-[10px] uppercase text-slate-400">
              Epoch Mag
            </span>
            <span className="font-mono text-purple-200">
              {currentEpoch
                ? `${currentEpoch.mag.toFixed(2)} +/- ${currentEpoch.mag_err.toFixed(2)}`
                : "--"}
            </span>
          </div>
        </div>
      </div>

      {/* Transport & Cadence Controls */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            aria-label="Previous epoch"
            onClick={handleStepPrev}
            disabled={epochCount === 0}
            className="rounded border border-[#3b185f] bg-[#181b28] px-2.5 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-[#581c87] hover:bg-[#3b185f] disabled:opacity-40"
          >
            Prev
          </button>

          <button
            type="button"
            aria-label={isPlaying ? "Pause blink loop" : "Play blink loop"}
            onClick={() => setIsPlaying((prev) => !prev)}
            disabled={epochCount === 0}
            className="min-w-[74px] rounded border border-[#581c87] bg-[#3b185f] px-3 py-1.5 text-xs font-semibold text-purple-100 transition-colors hover:bg-[#581c87] disabled:opacity-40"
          >
            {isPlaying ? "Pause" : "Play"}
          </button>

          <button
            type="button"
            aria-label="Next epoch"
            onClick={handleStepNext}
            disabled={epochCount === 0}
            className="rounded border border-[#3b185f] bg-[#181b28] px-2.5 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-[#581c87] hover:bg-[#3b185f] disabled:opacity-40"
          >
            Next
          </button>
        </div>

        {/* Direct Epoch Frame Selector with Survey Tags */}
        {cutoutSequence && cutoutSequence.epochs.length > 0 && (
          <div className="flex flex-wrap items-center gap-1">
            {cutoutSequence.epochs.map((ep, idx) => {
              const isActive = idx === safeEpochIndex;
              const epSurvey = ep.survey_id ?? cutoutSequence.survey_id ?? "ZTF";
              const epBadge = getSurveyBadgeDetails(epSurvey);
              const epBand = ep.filter_band ?? "r";
              return (
                <button
                  key={ep.epoch_index}
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    onEpochChange(idx);
                  }}
                  className={`rounded border px-2 py-1 text-xs font-mono transition-colors ${
                    isActive
                      ? "border-[#581c87] bg-[#581c87] font-semibold text-white shadow-sm"
                      : "border-[#26293d] bg-[#090a0f] text-slate-400 hover:border-[#3b185f] hover:text-slate-200"
                  }`}
                  title={`${epBadge.label} (${epBand})`}
                >
                  E{idx + 1}
                  {isMultiSurvey && (
                    <span className="ml-1 text-[10px] opacity-75">
                      {epBadge.shortName}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Stretch, Palette, Cadence, and Overlay Controls */}
      <div className="mt-3 grid grid-cols-2 gap-2 border-t border-[#26293d] pt-3 text-xs sm:grid-cols-3">
        <label className="flex flex-col gap-1 text-slate-400">
          <span className="text-[10px] uppercase tracking-wider">Cadence</span>
          <select
            aria-label="Blink cadence"
            value={cadenceHz}
            onChange={(e) => setCadenceHz(Number(e.target.value))}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-2 py-1 text-xs text-slate-200 focus:border-[#581c87] focus:outline-none"
          >
            {CADENCE_OPTIONS_HZ.map((hz) => (
              <option key={hz} value={hz}>
                {hz} Hz ({Math.round(1000 / hz)} ms)
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-slate-400">
          <span className="text-[10px] uppercase tracking-wider">Stretch</span>
          <select
            aria-label="Pixel stretch algorithm"
            value={stretchMode}
            onChange={(e) => setStretchMode(e.target.value as StretchMode)}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-2 py-1 text-xs text-slate-200 focus:border-[#581c87] focus:outline-none"
          >
            <option value="zscale">ZScale Clip</option>
            <option value="sqrt">Square Root</option>
            <option value="linear">Linear Min/Max</option>
          </select>
        </label>

        <label className="col-span-2 flex flex-col gap-1 text-slate-400 sm:col-span-1">
          <span className="text-[10px] uppercase tracking-wider">Palette</span>
          <select
            aria-label="Cutout color palette"
            value={colorMap}
            onChange={(e) => setColorMap(e.target.value as ColorMapMode)}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-2 py-1 text-xs text-slate-200 focus:border-[#581c87] focus:outline-none"
          >
            <option value="violet">Observatory Violet</option>
            <option value="grayscale">FITS Grayscale</option>
            <option value="inverted">Inverted Print</option>
          </select>
        </label>
      </div>

      <div className="mt-2.5 flex flex-wrap items-center gap-4 text-xs text-slate-300">
        <label className="inline-flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={showReticle}
            onChange={(e) => setShowReticle(e.target.checked)}
            className="accent-[#581c87]"
          />
          <span>Target Reticle</span>
        </label>

        <label className="inline-flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={showMotionTrack}
            onChange={(e) => setShowMotionTrack(e.target.checked)}
            className="accent-[#581c87]"
          />
          <span>Trajectory Vector</span>
        </label>

        <label className="inline-flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={normalizeFov}
            onChange={(e) => setNormalizeFov(e.target.checked)}
            className="accent-[#581c87]"
          />
          <span title="Align physical field of view scale across disparate instruments to eliminate spatial jumping">
            Normalize FOV (Scale Align)
          </span>
        </label>

        <label className="inline-flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={showScaleBar}
            onChange={(e) => setShowScaleBar(e.target.checked)}
            className="accent-[#581c87]"
          />
          <span>Scale Bar</span>
        </label>
      </div>
    </section>
  );
};

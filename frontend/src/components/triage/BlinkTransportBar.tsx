import React from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import type { ColorMapMode, StretchMode } from "../../types/api";

export interface BlinkTransportBarProps {
  readonly isPlaying: boolean;
  readonly cadenceHz: number;
  readonly stretchMode: StretchMode;
  readonly colorMap: ColorMapMode;
  readonly diffColorMap: ColorMapMode;
  readonly showReticle: boolean;
  readonly epochCount: number;
  readonly activeEpochIndex: number;
  readonly onTogglePlay: () => void;
  readonly onStepPrev: () => void;
  readonly onStepNext: () => void;
  readonly onSelectEpoch: (idx: number) => void;
  readonly onChangeCadence: (hz: number) => void;
  readonly onChangeStretch: (stretch: StretchMode) => void;
  readonly onChangeColorMap: (map: ColorMapMode) => void;
  readonly onChangeDiffColorMap: (map: ColorMapMode) => void;
  readonly onToggleReticle: (val: boolean) => void;
}

const CADENCE_OPTIONS: readonly number[] = [0.5, 1, 2, 4, 8];

export const BlinkTransportBar: React.FC<BlinkTransportBarProps> = ({
  isPlaying,
  cadenceHz,
  stretchMode,
  colorMap,
  diffColorMap,
  showReticle,
  epochCount,
  activeEpochIndex,
  onTogglePlay,
  onStepPrev,
  onStepNext,
  onSelectEpoch,
  onChangeCadence,
  onChangeStretch,
  onChangeColorMap,
  onChangeDiffColorMap,
  onToggleReticle,
}) => {
  return (
    <div className="flex flex-col gap-2.5 rounded-lg border border-[#26293d] bg-[#11131c] p-3 text-xs">
      {/* Top Row: Play/Pause, Step & Epoch Pills */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#181b28] pb-2">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onStepPrev}
            disabled={epochCount <= 1}
            aria-label="Previous epoch"
            title="Previous Epoch ([ or ArrowLeft)"
            className="rounded border border-[#3b185f] bg-[#181b28] p-1.5 text-slate-200 hover:bg-[#3b185f] disabled:opacity-40"
          >
            <SkipBack className="h-3.5 w-3.5" />
          </button>

          <button
            type="button"
            onClick={onTogglePlay}
            disabled={epochCount <= 1}
            aria-label={isPlaying ? "Pause blink loop" : "Play blink loop"}
            title="Play / Pause (Space)"
            className="flex items-center gap-1.5 rounded border border-[#581c87] bg-[#3b185f] px-3 py-1 font-semibold text-purple-100 hover:bg-[#581c87] disabled:opacity-40"
          >
            {isPlaying ? (
              <>
                <Pause className="h-3.5 w-3.5" />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5" />
                <span>Blink</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={onStepNext}
            disabled={epochCount <= 1}
            aria-label="Next epoch"
            title="Next Epoch (] or ArrowRight)"
            className="rounded border border-[#3b185f] bg-[#181b28] p-1.5 text-slate-200 hover:bg-[#3b185f] disabled:opacity-40"
          >
            <SkipForward className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Epoch Selector Buttons */}
        {epochCount > 0 && (
          <div className="flex items-center gap-1">
            {Array.from({ length: epochCount }, (_, i) => {
              const isActive = i === activeEpochIndex;
              return (
                <button
                  key={i}
                  type="button"
                  onClick={() => onSelectEpoch(i)}
                  className={`rounded px-2 py-0.5 font-mono text-[11px] font-semibold transition-colors ${
                    isActive
                      ? "border border-[#581c87] bg-[#581c87] text-white shadow-inner"
                      : "border border-[#26293d] bg-[#090a0f] text-slate-400 hover:border-[#3b185f] hover:text-slate-200"
                  }`}
                >
                  E{i + 1}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Bottom Row: Cadence, Stretch, Palette & Reticle Options */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-5 text-slate-300">
        <label className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase text-slate-400">Cadence</span>
          <select
            value={cadenceHz}
            onChange={(e) => onChangeCadence(Number(e.target.value))}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-1.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-[#581c87]"
          >
            {CADENCE_OPTIONS.map((hz) => (
              <option key={hz} value={hz}>
                {hz} Hz ({Math.round(1000 / hz)} ms)
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase text-slate-400">Stretch</span>
          <select
            value={stretchMode}
            onChange={(e) => onChangeStretch(e.target.value as StretchMode)}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-1.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-[#581c87]"
          >
            <option value="zscale">ZScale (Median)</option>
            <option value="asinh">Asinh</option>
            <option value="sqrt">Square Root</option>
            <option value="linear">Linear</option>
          </select>
        </label>

        <label className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase text-slate-400">Sci/Ref Palette</span>
          <select
            value={colorMap}
            onChange={(e) => onChangeColorMap(e.target.value as ColorMapMode)}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-1.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-[#581c87]"
          >
            <option value="violet">Observatory Violet</option>
            <option value="grayscale">Grayscale</option>
            <option value="inverted">Inverted</option>
          </select>
        </label>

        <label className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase text-slate-400">Diff Palette</span>
          <select
            value={diffColorMap}
            onChange={(e) => onChangeDiffColorMap(e.target.value as ColorMapMode)}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-1.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-[#581c87]"
          >
            <option value="zogy_diverging">ZOGY Diverging</option>
            <option value="violet">Observatory Violet</option>
            <option value="grayscale">Grayscale</option>
          </select>
        </label>

        <div className="flex items-end pb-1 col-span-2 sm:col-span-1">
          <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={showReticle}
              onChange={(e) => onToggleReticle(e.target.checked)}
              className="accent-[#581c87]"
            />
            <span>Target Reticle</span>
          </label>
        </div>
      </div>
    </div>
  );
};

import React, { useEffect, useState } from "react";
import type { CandidateCutoutSequenceResponse, ColorMapMode, StretchMode } from "../../types/api";
import { SingleChannelStamp } from "./SingleChannelStamp";
import { BlinkTransportBar } from "./BlinkTransportBar";

export interface ThreeChannelBlinkViewerProps {
  readonly cutoutSequence: CandidateCutoutSequenceResponse | null;
  readonly isLoading: boolean;
  readonly activeEpochIndex: number;
  readonly onEpochChange: (nextIndex: number) => void;
}

export const ThreeChannelBlinkViewer: React.FC<ThreeChannelBlinkViewerProps> = ({
  cutoutSequence,
  isLoading,
  activeEpochIndex,
  onEpochChange,
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [cadenceHz, setCadenceHz] = useState<number>(2);
  const [stretchMode, setStretchMode] = useState<StretchMode>("zscale");
  const [colorMap, setColorMap] = useState<ColorMapMode>("violet");
  const [diffColorMap, setDiffColorMap] = useState<ColorMapMode>("zogy_diverging");
  const [showReticle, setShowReticle] = useState<boolean>(true);

  const epochs = cutoutSequence?.epochs ?? [];
  const epochCount = epochs.length;
  const safeEpochIndex = epochCount > 0 ? Math.min(activeEpochIndex, epochCount - 1) : 0;
  const currentEpoch = epochs[safeEpochIndex] ?? null;

  // Animation blink timer
  useEffect(() => {
    if (!isPlaying || epochCount <= 1) {
      return undefined;
    }
    const intervalMs = Math.max(80, Math.round(1000 / cadenceHz));
    const timerId = setInterval(() => {
      onEpochChange((safeEpochIndex + 1) % epochCount);
    }, intervalMs);

    return () => clearInterval(timerId);
  }, [isPlaying, cadenceHz, epochCount, safeEpochIndex, onEpochChange]);

  const handleStepPrev = (): void => {
    if (epochCount <= 1) return;
    onEpochChange((safeEpochIndex - 1 + epochCount) % epochCount);
  };

  const handleStepNext = (): void => {
    if (epochCount <= 1) return;
    onEpochChange((safeEpochIndex + 1) % epochCount);
  };

  return (
    <section
      aria-labelledby="three-channel-blink-heading"
      className="flex flex-col gap-3 rounded-lg border border-[#3b185f] bg-[#11131c] p-3.5 shadow-lg shadow-black/40"
    >
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#26293d] pb-2">
        <div>
          <h2 id="three-channel-blink-heading" className="text-xs font-bold uppercase tracking-wider text-slate-100">
            Multi-Epoch 3-Channel Blink Matrix
          </h2>
          <p className="text-[11px] text-slate-400">
            {cutoutSequence
              ? `Target: ${cutoutSequence.candidate_id} | Field: ${cutoutSequence.field_id} (${cutoutSequence.cutout_size}x${cutoutSequence.cutout_size} px)`
              : "Select a candidate to inspect multi-epoch cutouts"}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {isLoading && (
            <span className="text-[10px] font-mono text-purple-300 animate-pulse">
              Loading Cutouts...
            </span>
          )}
          <div className="rounded border border-[#581c87] bg-[#3b185f]/50 px-2 py-0.5 text-xs font-mono font-semibold text-purple-200 tabular-nums">
            {epochCount > 0 ? `Epoch ${safeEpochIndex + 1} / ${epochCount}` : "Epoch 0 / 0"}
          </div>
        </div>
      </div>

      {/* 3-Channel Side-by-Side Canvas Viewport */}
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        <SingleChannelStamp
          channelLabel="1. Science Channel"
          channelSub="Raw Exposure"
          pixels={currentEpoch?.pixels}
          cutoutSize={cutoutSequence?.cutout_size ?? 31}
          targetX={currentEpoch?.target_x_px ?? 15}
          targetY={currentEpoch?.target_y_px ?? 15}
          colorMap={colorMap}
          stretchMode={stretchMode}
          showReticle={showReticle}
          canvasSizePx={210}
        />

        <SingleChannelStamp
          channelLabel="2. Reference Template"
          channelSub="Coadded Sky"
          pixels={currentEpoch?.reference_pixels}
          cutoutSize={cutoutSequence?.cutout_size ?? 31}
          targetX={currentEpoch?.target_x_px ?? 15}
          targetY={currentEpoch?.target_y_px ?? 15}
          colorMap={colorMap}
          stretchMode={stretchMode}
          showReticle={false}
          canvasSizePx={210}
        />

        <SingleChannelStamp
          channelLabel="3. ZOGY Difference"
          channelSub="Subtracted PSF"
          pixels={currentEpoch?.difference_pixels}
          cutoutSize={cutoutSequence?.cutout_size ?? 31}
          targetX={currentEpoch?.target_x_px ?? 15}
          targetY={currentEpoch?.target_y_px ?? 15}
          colorMap={diffColorMap}
          stretchMode={stretchMode}
          showReticle={showReticle}
          canvasSizePx={210}
        />
      </div>

      {/* Telemetry Bar for Current Epoch */}
      {currentEpoch && (
        <div className="grid grid-cols-3 gap-2 rounded border border-[#26293d] bg-[#090a0f] p-2 text-xs tabular-nums font-mono text-slate-300">
          <div>
            <span className="block text-[10px] uppercase text-slate-400">BJD_TDB</span>
            <span>{currentEpoch.bjd_tdb.toFixed(5)}</span>
          </div>
          <div>
            <span className="block text-[10px] uppercase text-slate-400">Target Centroid (X,Y)</span>
            <span>{currentEpoch.target_x_px.toFixed(1)}, {currentEpoch.target_y_px.toFixed(1)} px</span>
          </div>
          <div>
            <span className="block text-[10px] uppercase text-slate-400">Epoch Mag</span>
            <span className="text-purple-200">
              {currentEpoch.mag.toFixed(2)} +/- {currentEpoch.mag_err.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {/* Transport Controls */}
      <BlinkTransportBar
        isPlaying={isPlaying}
        cadenceHz={cadenceHz}
        stretchMode={stretchMode}
        colorMap={colorMap}
        diffColorMap={diffColorMap}
        showReticle={showReticle}
        epochCount={epochCount}
        activeEpochIndex={safeEpochIndex}
        onTogglePlay={() => setIsPlaying((p) => !p)}
        onStepPrev={handleStepPrev}
        onStepNext={handleStepNext}
        onSelectEpoch={onEpochChange}
        onChangeCadence={setCadenceHz}
        onChangeStretch={setStretchMode}
        onChangeColorMap={setColorMap}
        onChangeDiffColorMap={setDiffColorMap}
        onToggleReticle={setShowReticle}
      />
    </section>
  );
};

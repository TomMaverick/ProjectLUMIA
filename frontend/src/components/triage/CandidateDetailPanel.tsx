import React from "react";
import type {
  CandidateCutoutSequenceResponse,
  CandidateDetection,
} from "../../types/api";
import {
  computePixelMotionRatePxPerHour,
  formatDecSexagesimal,
  formatRaSexagesimal,
} from "../../utils/astronomyFormat";

export interface CandidateDetailPanelProps {
  readonly candidate: CandidateDetection | null;
  readonly cutoutSequence: CandidateCutoutSequenceResponse | null;
  readonly activeEpochIndex: number;
  readonly onSelectEpoch: (epochIndex: number) => void;
}

const MPC_RESIDUAL_LIMIT_ARCSEC = 0.25;

export const CandidateDetailPanel: React.FC<CandidateDetailPanelProps> = ({
  candidate,
  cutoutSequence,
  activeEpochIndex,
  onSelectEpoch,
}) => {
  if (!candidate) {
    return (
      <aside className="rounded-lg border border-[#3b185f] bg-[#11131c] p-4 text-xs text-slate-400 shadow-md">
        Select a moving-object candidate from the triage queue to inspect Real/Bogus neural classification,
        astrometric residuals, and multi-epoch photometry.
      </aside>
    );
  }

  const cnnPercent = (candidate.cnn_score * 100).toFixed(1);
  const bogusPercent = ((1.0 - candidate.cnn_score) * 100).toFixed(1);
  const classificationTier =
    candidate.cnn_score >= 0.8
      ? "HIGH CONFIDENCE REAL"
      : candidate.cnn_score >= 0.5
        ? "AMBIGUOUS / MANUAL REVIEW"
        : "LOW SCORE / LIKELY BOGUS";

  const withinMpcLimit = candidate.astrometric_residual_arcsec <= MPC_RESIDUAL_LIMIT_ARCSEC;

  const epochs = cutoutSequence?.epochs ?? [];
  const motionRatePxHr = computePixelMotionRatePxPerHour(epochs);

  const mags = epochs.map((ep) => ep.mag);
  const minMag = mags.length > 0 ? Math.min(...mags) : candidate.mag - 0.1;
  const maxMag = mags.length > 0 ? Math.max(...mags) : candidate.mag + 0.1;
  const amplitudeMag = mags.length > 1 ? maxMag - minMag : 0.0;

  // SVG Light Curve Geometry
  const svgW = 340;
  const svgH = 110;
  const padX = 32;
  const padY = 16;
  const plotW = svgW - padX * 2;
  const plotH = svgH - padY * 2;

  const magPad = Math.max(0.08, (maxMag - minMag) * 0.4);
  const yBright = minMag - magPad;
  const yFaint = maxMag + magPad;
  const ySpan = Math.max(1e-4, yFaint - yBright);

  const firstBjd = epochs[0]?.bjd_tdb ?? candidate.bjd_tdb;
  const lastBjd = epochs[epochs.length - 1]?.bjd_tdb ?? firstBjd + 0.03;
  const bjdSpan = Math.max(1e-6, lastBjd - firstBjd);

  return (
    <aside
      aria-labelledby="candidate-detail-heading"
      className="flex flex-col gap-3 rounded-lg border border-[#3b185f] bg-[#11131c] p-3.5 shadow-lg shadow-black/40"
    >
      {/* Header */}
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[#26293d] pb-2">
        <h2 id="candidate-detail-heading" className="text-xs font-bold uppercase tracking-wider text-slate-100">
          Target Diagnostics: <span className="font-mono text-purple-300">{candidate.candidate_id}</span>
        </h2>
        <span className="text-[11px] font-mono text-slate-400">
          Field {candidate.field_id} | BJD {candidate.bjd_tdb.toFixed(5)}
        </span>
      </div>

      {/* 1. CNN Score & Model Tag */}
      <div className="rounded border border-[#3b185f]/80 bg-[#090a0f] p-2.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold uppercase tracking-wider text-slate-300 text-[10px]">
            Real/Bogus Neural Classifier
          </span>
          <span className="rounded border border-[#581c87] bg-[#3b185f]/40 px-1.5 py-0.5 text-[9px] font-semibold tracking-wider text-purple-200">
            {classificationTier}
          </span>
        </div>

        <div className="mt-1.5 flex items-baseline justify-between tabular-nums">
          <div>
            <span className="text-xl font-bold text-purple-200">{cnnPercent}%</span>
            <span className="ml-1.5 text-[11px] text-slate-400">Real</span>
          </div>
          <div className="text-[11px] text-slate-400">
            Bogus residual: {bogusPercent}%
          </div>
        </div>

        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-[#181b28]">
          <div
            className="h-full bg-gradient-to-r from-[#3b185f] via-[#581c87] to-purple-400"
            style={{ width: `${Math.round(candidate.cnn_score * 100)}%` }}
          />
        </div>

        <div className="mt-1 text-[10px] text-slate-500 font-mono">
          Model: {candidate.model_version ?? "resnet_lite_s31_v2"}
        </div>
      </div>

      {/* 2. Astrometric Residuals & Coordinates */}
      <div className="grid grid-cols-2 gap-2 rounded border border-[#26293d] bg-[#090a0f] p-2.5 text-xs tabular-nums sm:grid-cols-4">
        <div>
          <span className="block text-[9px] uppercase tracking-wider text-slate-400">RA (ICRS)</span>
          <span className="mt-0.5 block font-mono font-semibold text-slate-100">
            {formatRaSexagesimal(candidate.ra_deg)}
          </span>
          <span className="font-mono text-[10px] text-slate-400">{candidate.ra_deg.toFixed(4)}°</span>
        </div>

        <div>
          <span className="block text-[9px] uppercase tracking-wider text-slate-400">Dec (ICRS)</span>
          <span className="mt-0.5 block font-mono font-semibold text-slate-100">
            {formatDecSexagesimal(candidate.dec_deg)}
          </span>
          <span className="font-mono text-[10px] text-slate-400">{candidate.dec_deg.toFixed(4)}°</span>
        </div>

        <div>
          <span className="block text-[9px] uppercase tracking-wider text-slate-400">Astrometric Res</span>
          <span
            className={`mt-0.5 block font-mono font-bold ${
              withinMpcLimit ? "text-emerald-300" : "text-amber-300"
            }`}
          >
            {candidate.astrometric_residual_arcsec.toFixed(2)}&quot;
          </span>
          <span className="text-[10px] text-slate-400">
            {withinMpcLimit ? "MPC Pass (<=0.25s)" : "Exceeds 0.25s"}
          </span>
        </div>

        <div>
          <span className="block text-[9px] uppercase tracking-wider text-slate-400">Motion Rate</span>
          <span className="mt-0.5 block font-mono font-semibold text-slate-100">
            {motionRatePxHr !== null ? `${motionRatePxHr.toFixed(1)} px/hr` : "--"}
          </span>
          <span className="text-[10px] text-slate-400">Linear fit</span>
        </div>
      </div>

      {/* 3. Photometric Light Curve */}
      <div className="rounded border border-[#26293d] bg-[#090a0f] p-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs tabular-nums mb-1.5">
          <span className="font-semibold uppercase tracking-wider text-slate-300 text-[10px]">
            Multi-Epoch Light Curve
          </span>
          <div className="flex items-center gap-2 text-[11px] text-slate-300">
            <span>Mag: <strong className="font-mono text-purple-200">{candidate.mag.toFixed(2)}</strong></span>
            <span>SNR: <strong className="font-mono text-slate-100">{candidate.snr.toFixed(1)}</strong></span>
            <span>Delta: <strong className="font-mono text-slate-100">{amplitudeMag.toFixed(2)}</strong></span>
          </div>
        </div>

        {epochs.length > 0 && (
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            role="img"
            aria-label="Multi-epoch photometric light curve"
            className="h-24 w-full rounded border border-[#181b28] bg-[#11131c]/70"
          >
            <line x1={padX} y1={padY} x2={svgW - padX} y2={padY} stroke="#26293d" strokeDasharray="3 3" />
            <line x1={padX} y1={padY + plotH / 2} x2={svgW - padX} y2={padY + plotH / 2} stroke="#26293d" strokeDasharray="3 3" />
            <line x1={padX} y1={svgH - padY} x2={svgW - padX} y2={svgH - padY} stroke="#26293d" strokeDasharray="3 3" />

            <text x={4} y={padY + 3} fill="#94a3b8" fontSize="8" fontFamily="monospace">
              {yBright.toFixed(2)}
            </text>
            <text x={4} y={svgH - padY + 3} fill="#94a3b8" fontSize="8" fontFamily="monospace">
              {yFaint.toFixed(2)}
            </text>

            {epochs.length > 1 && (
              <polyline
                fill="none"
                stroke="#7e22ce"
                strokeWidth="1.5"
                points={epochs
                  .map((ep, idx) => {
                    const x =
                      epochs.length === 1
                        ? padX + plotW / 2
                        : padX + ((ep.bjd_tdb - firstBjd) / bjdSpan) * plotW ||
                          padX + (idx / (epochs.length - 1)) * plotW;
                    const y = padY + ((ep.mag - yBright) / ySpan) * plotH;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(" ")}
              />
            )}

            {epochs.map((ep, idx) => {
              const cx =
                epochs.length === 1
                  ? padX + plotW / 2
                  : padX + ((ep.bjd_tdb - firstBjd) / bjdSpan) * plotW ||
                    padX + (idx / Math.max(1, epochs.length - 1)) * plotW;
              const cy = padY + ((ep.mag - yBright) / ySpan) * plotH;
              const isActive = idx === activeEpochIndex;

              return (
                <g key={ep.epoch_index} onClick={() => onSelectEpoch(idx)} className="cursor-pointer">
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isActive ? 4.5 : 3.0}
                    fill={isActive ? "#c084fc" : "#581c87"}
                    stroke={isActive ? "#ffffff" : "#a855f7"}
                    strokeWidth="1.2"
                  />
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </aside>
  );
};

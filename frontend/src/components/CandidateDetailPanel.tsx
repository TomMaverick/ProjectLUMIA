import React from "react";
import type {
  CandidateCutoutSequenceResponse,
  CandidateDetection,
  TriageDisposition,
} from "../types/api";
import { getSurveyBadgeDetails } from "./BlinkLoopViewer";
import {
  computePixelMotionRatePxPerHour,
  formatDecSexagesimal,
  formatRaSexagesimal,
} from "../utils/astronomyFormat";

export interface CandidateDetailPanelProps {
  readonly candidate: CandidateDetection | null;
  readonly cutoutSequence: CandidateCutoutSequenceResponse | null;
  readonly activeEpochIndex: number;
  readonly disposition: TriageDisposition;
  readonly onSetDisposition: (
    candidateId: string,
    disposition: TriageDisposition,
  ) => void;
  readonly onSelectEpoch: (epochIndex: number) => void;
}

const MPC_RESIDUAL_LIMIT_ARCSEC = 0.25;

export const CandidateDetailPanel: React.FC<CandidateDetailPanelProps> = ({
  candidate,
  cutoutSequence,
  activeEpochIndex,
  disposition,
  onSetDisposition,
  onSelectEpoch,
}) => {
  if (!candidate) {
    return (
      <aside className="rounded-lg border border-[#3b185f] bg-[#11131c] p-5 text-xs text-slate-400 shadow-lg shadow-black/40">
        Select a moving-object candidate from the triage queue to inspect
        Real/Bogus neural classification, astrometric residuals, and multi-epoch
        photometry.
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

  const withinMpcLimit =
    candidate.astrometric_residual_arcsec <= MPC_RESIDUAL_LIMIT_ARCSEC;

  const epochs = cutoutSequence?.epochs ?? [];
  const motionRatePxHr = computePixelMotionRatePxPerHour(epochs);

  // Compute photometric statistics across cutout epochs
  const mags = epochs.map((ep) => ep.mag);
  const minMag = mags.length > 0 ? Math.min(...mags) : candidate.mag - 0.1;
  const maxMag = mags.length > 0 ? Math.max(...mags) : candidate.mag + 0.1;
  const amplitudeMag = mags.length > 1 ? maxMag - minMag : 0.0;

  // Light curve SVG geometry
  const svgW = 360;
  const svgH = 120;
  const padX = 36;
  const padY = 18;
  const plotW = svgW - padX * 2;
  const plotH = svgH - padY * 2;

  const magPad = Math.max(0.08, (maxMag - minMag) * 0.45);
  const yBright = minMag - magPad;
  const yFaint = maxMag + magPad;
  const ySpan = Math.max(1e-4, yFaint - yBright);

  const firstBjd = epochs[0]?.bjd_tdb ?? candidate.bjd_tdb;
  const lastBjd = epochs[epochs.length - 1]?.bjd_tdb ?? firstBjd + 0.03;
  const bjdSpan = Math.max(1e-6, lastBjd - firstBjd);

  return (
    <aside
      aria-labelledby="candidate-detail-heading"
      className="flex flex-col gap-4 rounded-lg border border-[#3b185f] bg-[#11131c] p-4 shadow-lg shadow-black/40"
    >
      {/* Header & Operator Disposition Actions */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#26293d] pb-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2
              id="candidate-detail-heading"
              className="text-sm font-semibold uppercase tracking-wider text-slate-100"
            >
              Candidate Diagnostics:{" "}
              <span className="font-mono text-purple-300">
                {candidate.candidate_id}
              </span>
            </h2>
            <span
              className={`rounded border px-2 py-0.5 text-xs font-semibold ${
                getSurveyBadgeDetails(candidate.survey_id).badgeClass
              }`}
            >
              {getSurveyBadgeDetails(candidate.survey_id).label}
            </span>
            <span className="rounded border border-[#581c87] bg-[#22123b] px-1.5 py-0.5 font-mono text-xs font-medium text-purple-200">
              band: {candidate.filter_band ?? "r"}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Field {candidate.field_id} | BJD_TDB {candidate.bjd_tdb.toFixed(5)}
            {candidate.light_travel_time_sec !== undefined &&
              candidate.light_travel_time_sec !== null && (
                <span className="ml-2 font-mono text-cyan-300">
                  | Light-Travel Time: {candidate.light_travel_time_sec.toFixed(3)}s
                </span>
              )}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() =>
              onSetDisposition(candidate.candidate_id, "confirmed_real")
            }
            className={`rounded border px-2.5 py-1 text-xs font-semibold transition-colors ${
              disposition === "confirmed_real"
                ? "border-emerald-500 bg-emerald-900/70 text-emerald-100"
                : "border-[#3b185f] bg-[#181b28] text-slate-200 hover:border-emerald-600"
            }`}
          >
            Confirm Real
          </button>

          <button
            type="button"
            onClick={() =>
              onSetDisposition(candidate.candidate_id, "follow_up")
            }
            className={`rounded border px-2.5 py-1 text-xs font-semibold transition-colors ${
              disposition === "follow_up"
                ? "border-purple-400 bg-[#581c87] text-white"
                : "border-[#3b185f] bg-[#181b28] text-slate-200 hover:bg-[#3b185f]"
            }`}
          >
            Follow-Up
          </button>

          <button
            type="button"
            onClick={() =>
              onSetDisposition(candidate.candidate_id, "flagged_bogus")
            }
            className={`rounded border px-2.5 py-1 text-xs font-semibold transition-colors ${
              disposition === "flagged_bogus"
                ? "border-rose-500 bg-rose-900/70 text-rose-100"
                : "border-[#3b185f] bg-[#181b28] text-slate-200 hover:border-rose-600"
            }`}
          >
            Flag Bogus
          </button>
        </div>
      </div>

      {/* 1. CNN Real/Bogus Classification Card */}
      <div className="rounded-md border border-[#3b185f]/80 bg-[#090a0f] p-3.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold uppercase tracking-wider text-slate-300">
            Real/Bogus Neural Classifier
          </span>
          <span className="rounded border border-[#581c87] bg-[#3b185f]/40 px-2 py-0.5 text-[10px] font-semibold tracking-wider text-purple-200">
            {classificationTier}
          </span>
        </div>

        <div className="mt-2 flex items-baseline justify-between tabular-nums">
          <div>
            <span className="text-2xl font-bold text-purple-200">
              {cnnPercent}%
            </span>
            <span className="ml-2 text-xs text-slate-400">
              Real probability
            </span>
          </div>
          <div className="text-xs text-slate-400">
            Bogus residual: {bogusPercent}%
          </div>
        </div>

        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-[#181b28]">
          <div
            className="h-full bg-gradient-to-r from-[#3b185f] via-[#581c87] to-purple-400 transition-all"
            style={{ width: `${Math.round(candidate.cnn_score * 100)}%` }}
          />
        </div>
      </div>

      {/* 2. Astrometric Residuals & Sky Coordinates Card */}
      <div className="grid grid-cols-2 gap-3 rounded-md border border-[#26293d] bg-[#090a0f] p-3.5 text-xs tabular-nums sm:grid-cols-4">
        <div>
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">
            RA (ICRS)
          </span>
          <span className="mt-0.5 block font-mono font-semibold text-slate-100">
            {formatRaSexagesimal(candidate.ra_deg)}
          </span>
          <span className="font-mono text-[11px] text-slate-400">
            {candidate.ra_deg.toFixed(5)} deg
          </span>
        </div>

        <div>
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">
            Dec (ICRS)
          </span>
          <span className="mt-0.5 block font-mono font-semibold text-slate-100">
            {formatDecSexagesimal(candidate.dec_deg)}
          </span>
          <span className="font-mono text-[11px] text-slate-400">
            {candidate.dec_deg.toFixed(5)} deg
          </span>
        </div>

        <div>
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">
            Astrometric Residual
          </span>
          <span
            className={`mt-0.5 block font-mono text-sm font-bold ${
              withinMpcLimit ? "text-emerald-300" : "text-amber-300"
            }`}
          >
            {candidate.astrometric_residual_arcsec.toFixed(2)}&quot;
          </span>
          <span className="text-[11px] text-slate-400">
            {withinMpcLimit ? "MPC Pass (<=0.25s)" : "Exceeds 0.25s"}
          </span>
        </div>

        <div>
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">
            Centroid Velocity
          </span>
          <span className="mt-0.5 block font-mono font-semibold text-slate-100">
            {motionRatePxHr !== null
              ? `${motionRatePxHr.toFixed(1)} px/hr`
              : "--"}
          </span>
          <span className="text-[11px] text-slate-400">Linear stack fit</span>
        </div>
      </div>

      {/* 3. Photometric Measurements & Multi-Epoch Light Curve */}
      <div className="rounded-md border border-[#26293d] bg-[#090a0f] p-3.5">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs tabular-nums">
          <span className="font-semibold uppercase tracking-wider text-slate-300">
            Aperture Photometry &amp; Multi-Epoch Light Curve
          </span>
          <div className="flex items-center gap-3 text-slate-300">
            <span>
              Stack Mag:{" "}
              <strong className="font-mono text-purple-200">
                {candidate.mag.toFixed(2)}
              </strong>
            </span>
            <span>
              SNR:{" "}
              <strong className="font-mono text-slate-100">
                {candidate.snr.toFixed(1)}
              </strong>
            </span>
            <span>
              Delta Mag:{" "}
              <strong className="font-mono text-slate-100">
                {amplitudeMag.toFixed(2)}
              </strong>
            </span>
          </div>
        </div>

        {epochs.length > 0 && (
          <div className="mt-3">
            <svg
              viewBox={`0 0 ${svgW} ${svgH}`}
              role="img"
              aria-label="Multi-epoch photometric light curve"
              className="h-28 w-full overflow-visible rounded border border-[#181b28] bg-[#11131c]/70"
            >
              {/* Horizontal reference grid lines (inverted magnitude axis: brighter on top) */}
              <line
                x1={padX}
                y1={padY}
                x2={svgW - padX}
                y2={padY}
                stroke="#26293d"
                strokeDasharray="3 3"
              />
              <line
                x1={padX}
                y1={padY + plotH / 2}
                x2={svgW - padX}
                y2={padY + plotH / 2}
                stroke="#26293d"
                strokeDasharray="3 3"
              />
              <line
                x1={padX}
                y1={svgH - padY}
                x2={svgW - padX}
                y2={svgH - padY}
                stroke="#26293d"
                strokeDasharray="3 3"
              />

              <text
                x={4}
                y={padY + 3}
                fill="#94a3b8"
                fontSize="9"
                fontFamily="monospace"
              >
                {yBright.toFixed(2)}
              </text>
              <text
                x={4}
                y={svgH - padY + 3}
                fill="#94a3b8"
                fontSize="9"
                fontFamily="monospace"
              >
                {yFaint.toFixed(2)}
              </text>

              {/* Polyline connecting epoch magnitudes */}
              {epochs.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#7e22ce"
                  strokeWidth="1.6"
                  points={epochs
                    .map((ep, idx) => {
                      const x =
                        epochs.length === 1
                          ? padX + plotW / 2
                          : padX +
                            ((ep.bjd_tdb - firstBjd) / bjdSpan) * plotW ||
                            padX + (idx / (epochs.length - 1)) * plotW;
                      const y = padY + ((ep.mag - yBright) / ySpan) * plotH;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(" ")}
                />
              )}

              {/* Epoch points & 1-sigma photometric error bars */}
              {epochs.map((ep, idx) => {
                const cx =
                  epochs.length === 1
                    ? padX + plotW / 2
                    : padX + ((ep.bjd_tdb - firstBjd) / bjdSpan) * plotW ||
                      padX + (idx / Math.max(1, epochs.length - 1)) * plotW;
                const cy = padY + ((ep.mag - yBright) / ySpan) * plotH;
                const yErrTop =
                  padY + ((ep.mag - ep.mag_err - yBright) / ySpan) * plotH;
                const yErrBot =
                  padY + ((ep.mag + ep.mag_err - yBright) / ySpan) * plotH;
                const isActive = idx === activeEpochIndex;

                return (
                  <g
                    key={ep.epoch_index}
                    onClick={() => onSelectEpoch(idx)}
                    className="cursor-pointer"
                  >
                    <line
                      x1={cx}
                      y1={Math.max(6, yErrTop)}
                      x2={cx}
                      y2={Math.min(svgH - 6, yErrBot)}
                      stroke={isActive ? "#e9d5ff" : "#9333ea"}
                      strokeWidth="1.4"
                    />
                    <circle
                      cx={cx}
                      cy={cy}
                      r={isActive ? 4.8 : 3.4}
                      fill={isActive ? "#c084fc" : "#581c87"}
                      stroke={isActive ? "#ffffff" : "#a855f7"}
                      strokeWidth="1.2"
                    />
                  </g>
                );
              })}
            </svg>
          </div>
        )}
      </div>
    </aside>
  );
};

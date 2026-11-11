import React from "react";
import type {
  CandidateDetection,
  CandidateFilterParams,
  TriageDisposition,
} from "../types/api";
import { getSurveyBadgeDetails } from "./BlinkLoopViewer";
import {
  formatDecSexagesimal,
  formatRaSexagesimal,
} from "../utils/astronomyFormat";

export interface CandidateTriageTableProps {
  readonly candidates: readonly CandidateDetection[];
  readonly selectedCandidateId: string | null;
  readonly dispositions: Readonly<Record<string, TriageDisposition>>;
  readonly filters: CandidateFilterParams;
  readonly onSelectCandidate: (candidateId: string) => void;
  readonly onFilterChange: (nextFilters: CandidateFilterParams) => void;
}

const DISPOSITION_BADGE: Record<
  TriageDisposition,
  { readonly label: string; readonly className: string }
> = {
  unreviewed: {
    label: "UNREVIEWED",
    className: "border-slate-700 bg-slate-900/70 text-slate-400",
  },
  confirmed_real: {
    label: "REAL",
    className: "border-emerald-700/80 bg-emerald-950/60 text-emerald-300",
  },
  follow_up: {
    label: "FOLLOW-UP",
    className: "border-[#581c87] bg-[#3b185f]/70 text-purple-200",
  },
  flagged_bogus: {
    label: "BOGUS",
    className: "border-rose-800/80 bg-rose-950/60 text-rose-300",
  },
};

export const CandidateTriageTable: React.FC<CandidateTriageTableProps> = ({
  candidates,
  selectedCandidateId,
  dispositions,
  filters,
  onSelectCandidate,
  onFilterChange,
}) => {
  const toggleSort = (
    column: CandidateFilterParams["sortBy"],
  ): void => {
    if (filters.sortBy === column) {
      onFilterChange({
        ...filters,
        sortDirection: filters.sortDirection === "desc" ? "asc" : "desc",
      });
    } else {
      onFilterChange({
        ...filters,
        sortBy: column,
        sortDirection:
          column === "mag" || column === "astrometric_residual_arcsec"
            ? "asc"
            : "desc",
      });
    }
  };

  return (
    <section
      aria-labelledby="triage-table-heading"
      className="flex flex-col rounded-lg border border-[#3b185f] bg-[#11131c] p-4 shadow-lg shadow-black/40"
    >
      {/* Header & Filter Controls */}
      <div className="flex flex-col gap-3 border-b border-[#26293d] pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2
            id="triage-table-heading"
            className="text-sm font-semibold uppercase tracking-wider text-slate-100"
          >
            Candidate Triage Queue
          </h2>
          <p className="text-xs text-slate-400">
            Showing {candidates.length} moving-object candidates ranked for
            operator verification
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-3 text-xs">
          <label className="flex flex-col gap-1 text-slate-300">
            <span className="text-[10px] uppercase tracking-wider text-slate-400">
              Min CNN Score: {filters.minCnnScore.toFixed(2)}
            </span>
            <input
              type="range"
              aria-label="Minimum CNN score filter"
              min={0}
              max={0.95}
              step={0.05}
              value={filters.minCnnScore}
              onChange={(e) =>
                onFilterChange({
                  ...filters,
                  minCnnScore: Number(e.target.value),
                })
              }
              className="w-32 accent-[#581c87]"
            />
          </label>

          <label className="flex flex-col gap-1 text-slate-300">
            <span className="text-[10px] uppercase tracking-wider text-slate-400">
              Field Filter
            </span>
            <input
              type="text"
              aria-label="Filter by field identifier"
              placeholder="All fields..."
              value={filters.fieldId}
              onChange={(e) =>
                onFilterChange({
                  ...filters,
                  fieldId: e.target.value,
                })
              }
              className="w-28 rounded border border-[#3b185f] bg-[#090a0f] px-2 py-1 text-xs text-slate-100 placeholder:text-slate-500 focus:border-[#581c87] focus:outline-none"
            />
          </label>

          <label className="flex flex-col gap-1 text-slate-300">
            <span className="text-[10px] uppercase tracking-wider text-slate-400">
              Disposition
            </span>
            <select
              aria-label="Filter by operator disposition"
              value={filters.dispositionFilter}
              onChange={(e) =>
                onFilterChange({
                  ...filters,
                  dispositionFilter:
                    e.target.value as CandidateFilterParams["dispositionFilter"],
                })
              }
              className="rounded border border-[#3b185f] bg-[#090a0f] px-2 py-1 text-xs text-slate-100 focus:border-[#581c87] focus:outline-none"
            >
              <option value="all">All Statuses</option>
              <option value="unreviewed">Unreviewed</option>
              <option value="confirmed_real">Confirmed Real</option>
              <option value="follow_up">Follow-Up</option>
              <option value="flagged_bogus">Flagged Bogus</option>
            </select>
          </label>
        </div>
      </div>

      {/* Data Table */}
      <div className="mt-3 overflow-x-auto">
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="border-b border-[#26293d] text-[11px] uppercase tracking-wider text-slate-400">
              <th className="px-3 py-2.5 font-semibold">Candidate</th>
              <th className="px-3 py-2.5 font-semibold">Field</th>
              <th className="px-3 py-2.5 font-semibold">ICRS (RA / Dec)</th>
              <th className="px-3 py-2.5 font-semibold">
                <button
                  type="button"
                  onClick={() => toggleSort("cnn_score")}
                  className="inline-flex items-center gap-1 uppercase hover:text-purple-300"
                >
                  <span>CNN R/B</span>
                  {filters.sortBy === "cnn_score" && (
                    <span>{filters.sortDirection === "desc" ? "v" : "^"}</span>
                  )}
                </button>
              </th>
              <th className="px-3 py-2.5 font-semibold">
                <button
                  type="button"
                  onClick={() => toggleSort("snr")}
                  className="inline-flex items-center gap-1 uppercase hover:text-purple-300"
                >
                  <span>SNR</span>
                  {filters.sortBy === "snr" && (
                    <span>{filters.sortDirection === "desc" ? "v" : "^"}</span>
                  )}
                </button>
              </th>
              <th className="px-3 py-2.5 font-semibold">
                <button
                  type="button"
                  onClick={() => toggleSort("mag")}
                  className="inline-flex items-center gap-1 uppercase hover:text-purple-300"
                >
                  <span>Mag</span>
                  {filters.sortBy === "mag" && (
                    <span>{filters.sortDirection === "desc" ? "v" : "^"}</span>
                  )}
                </button>
              </th>
              <th className="px-3 py-2.5 font-semibold">
                <button
                  type="button"
                  onClick={() => toggleSort("astrometric_residual_arcsec")}
                  className="inline-flex items-center gap-1 uppercase hover:text-purple-300"
                >
                  <span>Residual</span>
                  {filters.sortBy === "astrometric_residual_arcsec" && (
                    <span>{filters.sortDirection === "desc" ? "v" : "^"}</span>
                  )}
                </button>
              </th>
              <th className="px-3 py-2.5 font-semibold">Triage</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-[#26293d]/80 tabular-nums">
            {candidates.length === 0 ? (
              <tr>
                <td
                  colSpan={8}
                  className="px-3 py-8 text-center text-xs text-slate-400"
                >
                  No candidate detections match the active filter criteria.
                </td>
              </tr>
            ) : (
              candidates.map((cand) => {
                const isSelected = cand.candidate_id === selectedCandidateId;
                const disposition: TriageDisposition =
                  dispositions[cand.candidate_id] ?? "unreviewed";
                const badge = DISPOSITION_BADGE[disposition];

                const scoreColorClass =
                  cand.cnn_score >= 0.8
                    ? "bg-[#7e22ce]"
                    : cand.cnn_score >= 0.5
                      ? "bg-amber-500"
                      : "bg-rose-500";

                return (
                  <tr
                    key={cand.candidate_id}
                    onClick={() => onSelectCandidate(cand.candidate_id)}
                    aria-selected={isSelected}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-[#3b185f]/55 text-white shadow-inner"
                        : "hover:bg-[#181b28] text-slate-200"
                    }`}
                  >
                    <td className="px-3 py-2.5 font-mono font-semibold text-purple-200">
                      <div className="flex items-center gap-1.5">
                        <span>{cand.candidate_id}</span>
                        <span
                          className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${
                            getSurveyBadgeDetails(cand.survey_id).badgeClass
                          }`}
                        >
                          {getSurveyBadgeDetails(cand.survey_id).shortName}
                        </span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-slate-300">
                      {cand.field_id}
                    </td>
                    <td className="px-3 py-2.5 font-mono text-slate-300">
                      {formatRaSexagesimal(cand.ra_deg)}{" "}
                      {formatDecSexagesimal(cand.dec_deg)}
                    </td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-[#090a0f]">
                          <div
                            className={`h-full ${scoreColorClass}`}
                            style={{
                              width: `${Math.round(cand.cnn_score * 100)}%`,
                            }}
                          />
                        </div>
                        <span className="font-mono font-semibold">
                          {cand.cnn_score.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {cand.snr.toFixed(1)}
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {cand.mag.toFixed(2)}
                    </td>
                    <td className="px-3 py-2.5 font-mono text-slate-300">
                      {cand.astrometric_residual_arcsec.toFixed(2)} arcsec
                    </td>
                    <td className="px-3 py-2.5">
                      <span
                        className={`inline-block rounded border px-2 py-0.5 text-[10px] font-semibold tracking-wider ${badge.className}`}
                      >
                        {badge.label}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
};

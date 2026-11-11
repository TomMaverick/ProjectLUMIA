import React from "react";
import { Activity, Database, RefreshCw, Radio, Shield, User } from "lucide-react";
import type { ConnectionMode, DetectionSummaryResponse, PipelineStatusResponse } from "../../types/api";
import type { WsConnectionStatus } from "../../services/wsJobsClient";

export interface TopTelemetryHeaderProps {
  readonly status: PipelineStatusResponse | null;
  readonly summary: DetectionSummaryResponse | null;
  readonly connectionMode: ConnectionMode;
  readonly wsStatus: WsConnectionStatus;
  readonly isRefreshing: boolean;
  readonly lastUpdatedIso: string | null;
  readonly authEnabled?: boolean;
  readonly operatorUsername?: string | null;
  readonly onRefresh: () => void;
  readonly onReconnectWs?: () => void;
  readonly onResetDuckDb?: () => void;
}

export const TopTelemetryHeader: React.FC<TopTelemetryHeaderProps> = ({
  status,
  summary,
  connectionMode,
  wsStatus,
  isRefreshing,
  lastUpdatedIso: _lastUpdatedIso,
  authEnabled = false,
  operatorUsername = null,
  onRefresh,
  onReconnectWs,
  onResetDuckDb,
}) => {
  const [liveTimes, setLiveTimes] = React.useState<{ utc: string; local: string }>(() => {
    const d = new Date();
    return {
      utc: d.toISOString().slice(11, 19),
      local: `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`,
    };
  });

  React.useEffect(() => {
    const updateTimes = (): void => {
      const d = new Date();
      setLiveTimes({
        utc: d.toISOString().slice(11, 19),
        local: `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`,
      });
    };
    updateTimes();
    const interval = setInterval(updateTimes, 1000);
    return () => clearInterval(interval);
  }, []);

  const isOperational = status?.status === "operational" && connectionMode === "connected";

  const wsBadgeColor =
    wsStatus === "connected"
      ? "border-emerald-500/50 bg-emerald-950/40 text-emerald-300"
      : wsStatus === "reconnecting" || wsStatus === "connecting"
        ? "border-amber-500/50 bg-amber-950/40 text-amber-300"
        : "border-slate-700 bg-slate-900/60 text-slate-400";

  return (
    <header className="border-b border-[#3b185f]/80 bg-[#11131c] px-4 py-2.5 shadow-lg shadow-black/40 min-h-[3.5rem] h-auto">
      <div className="mx-auto flex flex-col gap-2.5 2xl:flex-row 2xl:items-center 2xl:justify-between max-w-[3600px]">
        {/* Left: Brand Identity & Engine Status */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2.5">
            <div
              aria-hidden="true"
              className="flex h-8 w-8 items-center justify-center rounded-lg border border-[#581c87] bg-[#3b185f]/50 shadow-inner shrink-0"
            >
              <Activity className="h-4 w-4 text-purple-300" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="font-semibold text-sm tracking-wide text-slate-100 shrink-0">
                  PROJECT LUMIA
                </h1>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded border border-[#581c87]/60 bg-[#24103a] text-purple-200 shrink-0">
                  Console v1.0
                </span>
              </div>
              <span className="text-[11px] text-slate-400 hidden sm:inline leading-tight whitespace-normal max-w-sm lg:max-w-md">
                Low-SNR Unidentified Moving Interplanetary Analyzer
              </span>
            </div>
          </div>

          <div className="h-5 w-px bg-[#26293d] hidden sm:block shrink-0" />

          {/* Operational State Pill */}
          <div
            role="status"
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-0.5 text-xs font-semibold tracking-wider shrink-0 ${
              connectionMode === "offline_simulation"
                ? "border-amber-700/70 bg-amber-950/50 text-amber-300"
                : isOperational
                  ? "border-emerald-600/70 bg-emerald-950/50 text-emerald-200"
                  : "border-rose-800/80 bg-rose-950/60 text-rose-300"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full shrink-0 ${
                connectionMode === "offline_simulation"
                  ? "bg-amber-400"
                  : isOperational
                    ? "bg-emerald-400 animate-pulse"
                    : "bg-rose-400"
              }`}
            />
            <span>
              {connectionMode === "offline_simulation"
                ? "OFFLINE SIMULATION"
                : isOperational
                  ? "OPERATIONAL"
                  : "DEGRADED"}
            </span>
          </div>

          {/* WebSocket Stream Badge */}
          <button
            type="button"
            onClick={onReconnectWs}
            title="Click to reconnect WebSocket stream"
            className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-mono cursor-pointer transition-opacity hover:opacity-80 shrink-0 ${wsBadgeColor}`}
          >
            <Radio className="h-3 w-3 shrink-0" />
            <span>Web Socket: {wsStatus.toUpperCase()}</span>
          </button>

          {/* DuckDB Status */}
          <button
            type="button"
            onClick={onResetDuckDb}
            title="Click to refresh / reset DuckDB analytical state"
            className="inline-flex items-center gap-1.5 rounded-md border border-[#0e7490]/60 bg-[#083344]/40 px-2 py-0.5 text-[11px] font-mono text-cyan-200 cursor-pointer transition-opacity hover:opacity-80 shrink-0"
          >
            <Database className="h-3 w-3 text-cyan-300 shrink-0" />
            <span>DUCKDB: {status?.duckdb_connected ? "OK" : "MOCK"}</span>
          </button>
        </div>

        {/* Right: Analytical Telemetry KPIs, Operator Badge & Refresh */}
        <div className="flex flex-wrap items-center gap-3 justify-between 2xl:justify-end">
          {/* Quick Metrics Strip */}
          <div className="flex items-center gap-3 text-xs tabular-nums text-slate-300 flex-wrap">
            <div className="hidden lg:block border-l border-[#26293d] pl-3">
              <span className="text-[10px] uppercase text-slate-400 block">Total Detections</span>
              <span className="font-semibold text-slate-100">{summary?.total_candidates ?? 0}</span>
            </div>

            <div className="border-l border-[#26293d] pl-3">
              <span className="text-[10px] uppercase text-purple-300 block">High Conf (&gt;=0.8)</span>
              <span className="font-semibold text-purple-200">{summary?.high_confidence_count ?? 0}</span>
            </div>

            <div className="hidden md:block border-l border-[#26293d] pl-3">
              <span className="text-[10px] uppercase text-slate-400 block">Mean SNR</span>
              <span className="font-semibold text-slate-100">
                {summary?.mean_snr !== null && summary?.mean_snr !== undefined
                  ? summary.mean_snr.toFixed(1)
                  : "--"}
              </span>
            </div>

            <div className="hidden xl:block border-l border-[#26293d] pl-3">
              <span className="text-[10px] uppercase text-slate-400 block">Sync UTC</span>
              <span className="font-mono text-slate-300">{liveTimes.utc}</span>
            </div>

            <div className="hidden xl:block border-l border-[#26293d] pl-3">
              <span className="text-[10px] uppercase text-slate-400 block">Local Time</span>
              <span className="font-mono text-slate-300">LOCAL {liveTimes.local}</span>
            </div>
          </div>

          {/* Operator Auth Badge */}
          {authEnabled ? (
            <div className="flex items-center gap-1.5 rounded-md border border-[#581c87]/70 bg-[#24103a]/70 px-2.5 py-1 text-xs text-purple-200">
              <User className="h-3.5 w-3.5 text-purple-300" />
              <span className="font-medium font-mono">{operatorUsername ?? "Operator"}</span>
            </div>
          ) : (
            <div className="hidden sm:inline-flex items-center gap-1 rounded-md border border-[#26293d] bg-[#181b28] px-2 py-1 text-[11px] text-slate-400 font-mono">
              <Shield className="h-3 w-3 text-amber-400" />
              <span>LOCAL BYPASS</span>
            </div>
          )}

          {/* Sync Button */}
          <button
            type="button"
            onClick={onRefresh}
            disabled={isRefreshing}
            className="inline-flex items-center gap-1.5 rounded-md border border-[#581c87] bg-[#3b185f] px-3 py-1 text-xs font-medium text-purple-100 transition-colors hover:bg-[#581c87] focus:outline-none focus:ring-2 focus:ring-[#a855f7] disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">{isRefreshing ? "Syncing..." : "Refresh"}</span>
          </button>
        </div>
      </div>
    </header>
  );
};

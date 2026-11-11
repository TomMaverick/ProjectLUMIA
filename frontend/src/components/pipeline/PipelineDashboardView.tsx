import React, { useState } from "react";
import { Plus, Radio } from "lucide-react";
import { TaskLogStreamConsole } from "./TaskLogStreamConsole";
import { JobSubmissionDrawer } from "./JobSubmissionDrawer";
import type { WsJobSummary, WsTaskStateChangedPayload, WsTaskLogAppendedPayload, WsJobMetricsUpdatedPayload } from "../../types/api";
import type { JobSubmissionRequest } from "../../types/settings";
import type { WsConnectionStatus } from "../../services/wsJobsClient";

const STAGE_FUNCTIONAL_LABELS: Record<number, string> = {
  1: "Optics / Config",
  2: "Catalog Ingest",
  3: "Astrometry & WCS",
  4: "ZOGY Subtraction",
  5: "Tracklet Linking",
  6: "Active Learning",
  7: "Neural Classifier",
  8: "Synthetic Injection",
  9: "MPC Ephemerides",
  10: "API & Triage",
};

export interface PipelineDashboardViewProps {
  readonly wsStatus: WsConnectionStatus;
  readonly activeJobs: readonly WsJobSummary[];
  readonly tasks: readonly WsTaskStateChangedPayload[];
  readonly logs: readonly WsTaskLogAppendedPayload[];
  readonly latestMetrics: WsJobMetricsUpdatedPayload | null;
  readonly droppedLogFrames?: number;
  readonly onSelectJob?: (jobId: string) => void;
  readonly onSubmitJob: (request: JobSubmissionRequest) => Promise<void>;
}

export const PipelineDashboardView: React.FC<PipelineDashboardViewProps> = ({
  wsStatus,
  activeJobs,
  tasks,
  logs,
  latestMetrics,
  droppedLogFrames = 0,
  onSubmitJob,
}) => {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(activeJobs[0]?.job_id ?? null);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);

  const selectedJob = activeJobs.find((j) => j.job_id === selectedJobId) ?? activeJobs[0] ?? null;

  return (
    <div className="flex flex-col gap-4 max-w-[1600px] mx-auto pb-8 text-xs">
      {/* Top Banner: WebSocket status & Job Launcher Action */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#26293d] pb-3">
        <div>
          <h2 className="text-base font-bold text-slate-100 uppercase tracking-wider">
            Live Pipeline Monitoring &amp; Telemetry
          </h2>
          <p className="text-slate-400">
            Real-time Prefect DAG execution, automated reduction pipeline stages, and DuckDB analytical extraction metrics
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 rounded border border-[#26293d] bg-[#11131c] px-2.5 py-1 font-mono text-[11px] text-slate-300">
            <Radio className="h-3 w-3 text-purple-400 animate-pulse" />
            <span>WS Stream: {wsStatus.toUpperCase()}</span>
          </div>

          <button
            type="button"
            onClick={() => setIsDrawerOpen(true)}
            className="inline-flex items-center gap-1.5 rounded border border-[#581c87] bg-[#581c87] px-3.5 py-1.5 font-bold text-white hover:bg-[#7e22ce] transition-colors shadow-md"
          >
            <Plus className="h-4 w-4" />
            <span>Submit Pipeline Job</span>
          </button>
        </div>
      </div>

      {/* KPI Cards: Analytical Extraction Metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded border border-[#26293d] bg-[#11131c] p-3 shadow-inner">
          <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Tracked Prefect Jobs</span>
          <span className="text-lg font-bold font-mono text-purple-200 tabular-nums">{activeJobs.length}</span>
          <span className="text-[10px] text-slate-500 block">PostgreSQL OLTP</span>
        </div>

        <div className="rounded border border-[#0e7490]/80 bg-[#083344]/40 p-3 shadow-inner">
          <span className="text-[10px] uppercase text-cyan-300 block mb-0.5">Extracted Candidates</span>
          <span className="text-lg font-bold font-mono text-cyan-100 tabular-nums">
            {latestMetrics?.extracted_candidates ?? 312}
          </span>
          <span className="text-[10px] text-cyan-400/80 block">DuckDB Analytical Plane</span>
        </div>

        <div className="rounded border border-[#581c87]/80 bg-[#24103a]/50 p-3 shadow-inner">
          <span className="text-[10px] uppercase text-purple-300 block mb-0.5">CNN Passed (&gt;=0.80)</span>
          <span className="text-lg font-bold font-mono text-purple-100 tabular-nums">
            {latestMetrics?.cnn_passed_candidates ?? 29}
          </span>
          <span className="text-[10px] text-purple-300/80 block">Real/Bogus Neural Classifier</span>
        </div>

        <div className="rounded border border-[#26293d] bg-[#11131c] p-3 shadow-inner">
          <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Linked Tracklets</span>
          <span className="text-lg font-bold font-mono text-emerald-300 tabular-nums">
            {latestMetrics?.linked_tracklets ?? 4}
          </span>
          <span className="text-[10px] text-slate-500 block">HelioLinC Orbit Linker</span>
        </div>
      </div>

      {/* Two Column Layout: Jobs Table + Task Stages */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Left: Jobs Queue (5 cols) */}
        <div className="rounded-lg border border-[#3b185f] bg-[#11131c] p-3.5 lg:col-span-5 shadow-lg shadow-black/40">
          <div className="flex items-center justify-between border-b border-[#26293d] pb-2 mb-3">
            <h3 className="font-bold uppercase tracking-wider text-slate-200 text-xs">
              Prefect Pipeline Jobs ({activeJobs.length})
            </h3>
            <span className="text-[10px] font-mono text-slate-400">WS Live Subscriptions</span>
          </div>

          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {activeJobs.length === 0 ? (
              <div className="text-slate-500 italic p-3 text-center">No active pipeline jobs recorded.</div>
            ) : (
              activeJobs.map((job) => {
                const isSel = job.job_id === selectedJob?.job_id;
                return (
                  <button
                    key={job.job_id}
                    type="button"
                    onClick={() => setSelectedJobId(job.job_id)}
                    className={`w-full text-left rounded border p-2.5 transition-colors ${
                      isSel
                        ? "border-[#581c87] bg-[#24103a] text-purple-100 shadow-inner"
                        : "border-[#26293d] bg-[#090a0f] text-slate-300 hover:border-[#3b185f]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono font-bold">{job.job_id}</span>
                      <span
                        className={`px-1.5 py-0.5 rounded font-mono text-[9px] uppercase font-bold ${
                          job.status === "running"
                            ? "bg-purple-900/60 text-purple-200 border border-purple-600 animate-pulse"
                            : job.status === "completed"
                              ? "bg-emerald-950/60 text-emerald-300 border border-emerald-700"
                              : "bg-slate-900 text-slate-400 border border-slate-700"
                        }`}
                      >
                        {job.status}
                      </span>
                    </div>

                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Field: {job.target_field_id}</span>
                      <span>Tasks: {job.completed_tasks} / {job.total_tasks}</span>
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Right: Pipeline Task Matrix (7 cols) */}
        <div className="rounded-lg border border-[#3b185f] bg-[#11131c] p-3.5 lg:col-span-7 shadow-lg shadow-black/40">
          <div className="flex items-center justify-between border-b border-[#26293d] pb-2 mb-3">
            <h3 className="font-bold uppercase tracking-wider text-slate-200 text-xs">
              Pipeline Execution Stages (Job: {selectedJob?.job_id ?? "--"})
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Retry Policy: Max 3</span>
          </div>

          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {tasks.length === 0 ? (
              <div className="text-slate-500 italic p-3 text-center">No task run events received.</div>
            ) : (
              tasks.map((task) => (
                <div
                  key={task.task_run_id}
                  className="flex items-center justify-between rounded border border-[#26293d] bg-[#090a0f] p-2"
                >
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-[#181b28] border border-[#3b185f] px-1.5 py-0.5 font-mono text-[10px] text-purple-300 font-semibold">
                      {STAGE_FUNCTIONAL_LABELS[task.module_number] ?? `Stage ${task.module_number}`}
                    </span>
                    <div>
                      <span className="font-semibold block text-slate-200">{task.task_name}</span>
                      <span className="text-[10px] text-slate-500">Attempt {task.attempt_count} / {task.max_retries}</span>
                    </div>
                  </div>

                  <span
                    className={`px-2 py-0.5 rounded font-mono text-[10px] uppercase font-bold ${
                      task.status === "completed"
                        ? "text-emerald-300 bg-emerald-950/50 border border-emerald-700/60"
                        : task.status === "running"
                          ? "text-purple-200 bg-purple-950/60 border border-purple-600 animate-pulse"
                          : task.status === "retrying"
                            ? "text-amber-300 bg-amber-950/60 border border-amber-600"
                            : "text-slate-400 bg-slate-900 border border-slate-700"
                    }`}
                  >
                    {task.status}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Live Log Console Terminal */}
      <TaskLogStreamConsole logs={logs} droppedLogFrames={droppedLogFrames} />

      {/* Drawer */}
      <JobSubmissionDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        onSubmitJob={onSubmitJob}
      />
    </div>
  );
};

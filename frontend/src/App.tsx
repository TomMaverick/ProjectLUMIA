import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ObservatoryAppShell } from "./components/shell/ObservatoryAppShell";
import type { ActiveNavView } from "./components/shell/CollapsibleNavRail";
import { CandidateTriageWorkspace } from "./components/triage/CandidateTriageWorkspace";
import { PipelineDashboardView } from "./components/pipeline/PipelineDashboardView";
import { GlobalOptionsView } from "./components/settings/GlobalOptionsView";
import { WsJobsClient, type WsConnectionStatus } from "./services/wsJobsClient";
import {
  OFFLINE_SIMULATION_CANDIDATES,
  OFFLINE_SIMULATION_STATUS,
  buildOfflineSummary,
  fetchCandidates,
  fetchDetectionSummary,
  fetchPipelineStatus,
} from "./services/lumiaApi";
import type {
  CandidateDetection,
  CandidateFilterParams,
  ConnectionMode,
  DetectionSummaryResponse,
  PipelineStatusResponse,
  TriageDisposition,
  WsJobSummary,
  WsTaskStateChangedPayload,
  WsTaskLogAppendedPayload,
  WsJobMetricsUpdatedPayload,
} from "./types/api";
import type { GlobalProgramSettings, JobSubmissionRequest } from "./types/settings";
import { requestJson, setApiBaseUrl } from "./services/httpClient";

const INITIAL_FILTERS: CandidateFilterParams = {
  minCnnScore: 0.0,
  fieldId: "",
  dispositionFilter: "all",
  sortBy: "cnn_score",
  sortDirection: "desc",
};

const DEFAULT_SETTINGS: GlobalProgramSettings = {
  backend_connection: "local",
  remote_base_url: "",
  auth_enabled: false,
  archive_storage_path: "data/archives/",
  duckdb: {
    engine_mode: "in_memory_parquet",
    database_file_path: "data/olap/lumia_analytics.duckdb",
    parquet_root_dir: "data/olap/",
    read_only_ui_readers: true,
    memory_limit_gb: 8,
    worker_threads: 8,
  },
  postgres: {
    connection_source: "structured_params",
    host: "127.0.0.1",
    port: 5432,
    database: "lumia_oltp",
    username: "lumia_operator",
    has_password_configured: true,
    env_password_var: null,
    ssl_mode: "require",
    pool_size: 10,
    max_overflow: 20,
  },
  display: {
    ui_scale_percent: 100,
    default_cadence_hz: 2,
    default_stretch_mode: "zscale",
    default_colormap: "violet",
    default_diff_colormap: "zogy_diverging",
    auto_advance_on_triage: true,
    log_terminal_buffer_size: 5000,
  },
};

const INITIAL_DEMO_JOBS: readonly WsJobSummary[] = [
  {
    job_id: "job-20261108-01",
    target_field_id: "ZTF_0412",
    observation_night_utc: "2026-11-08",
    status: "running",
    matched_catalog_sources: 18420,
    brightest_catalog_mag_g: 9.41,
    completed_tasks: 3,
    total_tasks: 5,
    submitted_at_utc: "2026-11-08T12:18:10.000Z",
    started_at_utc: "2026-11-08T12:18:12.240Z",
    completed_at_utc: null,
  },
  {
    job_id: "job-20261108-02",
    target_field_id: "ZTF_0413",
    observation_night_utc: "2026-11-08",
    status: "queued",
    matched_catalog_sources: null,
    brightest_catalog_mag_g: null,
    completed_tasks: 0,
    total_tasks: 5,
    submitted_at_utc: "2026-11-08T12:30:00.000Z",
    started_at_utc: null,
    completed_at_utc: null,
  },
];

const INITIAL_DEMO_TASKS: readonly WsTaskStateChangedPayload[] = [
  {
    task_run_id: "task-01",
    job_id: "job-20261108-01",
    task_name: "ingest_fits_and_catalog",
    module_number: 2,
    status: "completed",
    attempt_count: 1,
    max_retries: 3,
    next_retry_at_utc: null,
    started_at_utc: "2026-11-08T12:18:12.240Z",
    finished_at_utc: "2026-11-08T12:18:14.050Z",
  },
  {
    task_run_id: "task-02",
    job_id: "job-20261108-01",
    task_name: "wcs_astrometry_photometry",
    module_number: 3,
    status: "completed",
    attempt_count: 1,
    max_retries: 3,
    next_retry_at_utc: null,
    started_at_utc: "2026-11-08T12:18:14.100Z",
    finished_at_utc: "2026-11-08T12:18:28.300Z",
  },
  {
    task_run_id: "task-03",
    job_id: "job-20261108-01",
    task_name: "zogy_subtraction_cutouts_gpu",
    module_number: 4,
    status: "completed",
    attempt_count: 1,
    max_retries: 3,
    next_retry_at_utc: null,
    started_at_utc: "2026-11-08T12:18:28.400Z",
    finished_at_utc: "2026-11-08T12:18:34.800Z",
  },
  {
    task_run_id: "task-04",
    job_id: "job-20261108-01",
    task_name: "classify_cutout_triplets_gpu",
    module_number: 7,
    status: "running",
    attempt_count: 1,
    max_retries: 3,
    next_retry_at_utc: null,
    started_at_utc: "2026-11-08T12:18:35.000Z",
    finished_at_utc: null,
  },
  {
    task_run_id: "task-05",
    job_id: "job-20261108-01",
    task_name: "heliolinc_tracklet_linking",
    module_number: 5,
    status: "pending",
    attempt_count: 0,
    max_retries: 3,
    next_retry_at_utc: null,
    started_at_utc: null,
    finished_at_utc: null,
  },
];

const INITIAL_DEMO_LOGS: readonly WsTaskLogAppendedPayload[] = [
  {
    log_id: 1,
    job_id: "job-20261108-01",
    task_run_id: "task-01",
    attempt_number: 1,
    severity: "INFO",
    event_code: "CATALOG_INGESTED",
    message: "Queried Gaia DR3 reference stars: 18,420 sources matched sky-box.",
    logged_at_utc: "2026-11-08T12:18:14.050Z",
  },
  {
    log_id: 2,
    job_id: "job-20261108-01",
    task_run_id: "task-02",
    attempt_number: 1,
    severity: "INFO",
    event_code: "WCS_SOLVED",
    message: "Astrometric solution achieved. Mean residual 0.08 arcsec.",
    logged_at_utc: "2026-11-08T12:18:28.300Z",
  },
  {
    log_id: 3,
    job_id: "job-20261108-01",
    task_run_id: "task-03",
    attempt_number: 1,
    severity: "INFO",
    event_code: "ZOGY_COMPLETE",
    message: "Extracted 312 sub-threshold cutout triplets across 4 observation epochs.",
    logged_at_utc: "2026-11-08T12:18:34.800Z",
  },
  {
    log_id: 4,
    job_id: "job-20261108-01",
    task_run_id: "task-04",
    attempt_number: 1,
    severity: "INFO",
    event_code: "CNN_BATCH_SCORED",
    message: "Scored 312 31x31 ZOGY triplets in 3 mini-batches (29 >= threshold 0.80).",
    logged_at_utc: "2026-11-08T12:18:36.100Z",
  },
];

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ActiveNavView>("triage");
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null);
  const [summary, setSummary] = useState<DetectionSummaryResponse | null>(null);
  const [allCandidates, setAllCandidates] = useState<readonly CandidateDetection[]>([]);
  const [connectionMode, setConnectionMode] = useState<ConnectionMode>("connected");
  const [connectionNotice, setConnectionNotice] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(true);
  const [lastUpdatedIso, setLastUpdatedIso] = useState<string | null>(null);
  const [wsStatus, setWsStatus] = useState<WsConnectionStatus>("disconnected");

  // Pipeline Real-Time Telemetry State
  const [activeJobs, setActiveJobs] = useState<readonly WsJobSummary[]>(INITIAL_DEMO_JOBS);
  const [tasks, setTasks] = useState<readonly WsTaskStateChangedPayload[]>(INITIAL_DEMO_TASKS);
  const [logs, setLogs] = useState<readonly WsTaskLogAppendedPayload[]>(INITIAL_DEMO_LOGS);
  const [latestMetrics, setLatestMetrics] = useState<WsJobMetricsUpdatedPayload | null>(null);
  const [droppedLogFrames, setDroppedLogFrames] = useState<number>(0);

  // Global Settings State
  const [settings, setSettings] = useState<GlobalProgramSettings>(() => {
    try {
      const stored = localStorage.getItem("lumia.ui.settings.v1");
      return stored ? (JSON.parse(stored) as GlobalProgramSettings) : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [filters, setFilters] = useState<CandidateFilterParams>(INITIAL_FILTERS);
  const [dispositions, setDispositions] = useState<Readonly<Record<string, TriageDisposition>>>({});
  const wsClientRef = React.useRef<WsJobsClient | null>(null);

  // Sync API Base URL whenever backend_connection or remote_base_url changes
  useEffect(() => {
    if (settings.backend_connection === "remote" && settings.remote_base_url) {
      setApiBaseUrl(settings.remote_base_url);
    } else {
      setApiBaseUrl("");
    }
  }, [settings.backend_connection, settings.remote_base_url]);

  // Apply root UI viewport scale to HTML root so Tailwind rem units scale proportionally
  useEffect(() => {
    const scale = settings.display.ui_scale_percent ?? 100;
    document.documentElement.style.fontSize = `${scale}%`;
  }, [settings.display.ui_scale_percent]);

  // Initialize and connect WebSocket client for /api/v1/ws/jobs
  useEffect(() => {
    const wsClient = new WsJobsClient({ authEnabled: settings.auth_enabled });
    wsClientRef.current = wsClient;
    const unsubStatus = wsClient.addStatusListener(setWsStatus);
    const unsubEvents = wsClient.addEventListener((envelope) => {
      if (envelope.event_type === "snapshot") {
        const snap = envelope.payload as { active_jobs: WsJobSummary[] };
        if (snap.active_jobs) {
          setActiveJobs(snap.active_jobs);
        }
      } else if (envelope.event_type === "task.state_changed") {
        const changedTask = envelope.payload as WsTaskStateChangedPayload;
        setTasks((prev) => {
          const idx = prev.findIndex((t) => t.task_run_id === changedTask.task_run_id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = changedTask;
            return next;
          }
          return [...prev, changedTask];
        });
      } else if (envelope.event_type === "task.log_appended") {
        const newLog = envelope.payload as WsTaskLogAppendedPayload;
        setLogs((prev) => [...prev.slice(-settings.display.log_terminal_buffer_size), newLog]);
      } else if (envelope.event_type === "job.metrics_updated") {
        setLatestMetrics(envelope.payload as WsJobMetricsUpdatedPayload);
      } else if (envelope.event_type === "heartbeat") {
        const hb = envelope.payload as { dropped_log_frames: number };
        setDroppedLogFrames(hb.dropped_log_frames ?? 0);
      }
    });

    void wsClient.connect();

    return () => {
      unsubStatus();
      unsubEvents();
      wsClient.disconnect();
      if (wsClientRef.current === wsClient) {
        wsClientRef.current = null;
      }
    };
  }, [settings.auth_enabled, settings.backend_connection, settings.remote_base_url, settings.display.log_terminal_buffer_size]);

  const loadDashboardTelemetry = useCallback(async (signal?: AbortSignal) => {
    setIsRefreshing(true);
    try {
      const statusRes = await fetchPipelineStatus(signal);
      if (signal?.aborted) return;

      const [summaryRes, candidatesRes] = await Promise.all([
        fetchDetectionSummary(0.8, signal).catch(() => null),
        fetchCandidates({ limit: 200, signal }).catch(() => null),
      ]);
      if (signal?.aborted) return;

      setStatus(statusRes);
      if (statusRes.status === "operational" || statusRes.duckdb_connected) {
        setConnectionMode("connected");
        setConnectionNotice(null);
      }
      if (summaryRes) {
        setSummary(summaryRes);
      }
      if (candidatesRes) {
        setAllCandidates(candidatesRes.candidates);
        const targetCandidate = candidatesRes.candidates[0] ?? null;
        setSelectedCandidateId((prevId) => {
          if (prevId && candidatesRes.candidates.some((c) => c.candidate_id === prevId)) {
            return prevId;
          }
          return targetCandidate?.candidate_id ?? null;
        });
      }
      setLastUpdatedIso(new Date().toISOString().slice(11, 19));
    } catch {
      if (signal?.aborted) return;

      const fallbackTarget = OFFLINE_SIMULATION_CANDIDATES[0] ?? null;
      setConnectionMode("offline_simulation");
      setConnectionNotice(
        "FastAPI backend at /api/v1 is currently unreachable. Displaying synthetic moving-object preview telemetry.",
      );
      setStatus(OFFLINE_SIMULATION_STATUS);
      setSummary(buildOfflineSummary(OFFLINE_SIMULATION_CANDIDATES, 0.8));
      setAllCandidates(OFFLINE_SIMULATION_CANDIDATES);
      setLastUpdatedIso(new Date().toISOString().slice(11, 19));
      setSelectedCandidateId((prevId) => prevId ?? fallbackTarget?.candidate_id ?? null);
    } finally {
      if (!signal?.aborted) {
        setIsRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadDashboardTelemetry(controller.signal);
    return () => controller.abort();
  }, [loadDashboardTelemetry]);

  const filteredCandidates = useMemo(() => {
    const normalizedField = filters.fieldId.trim().toLowerCase();

    const filtered = allCandidates.filter((cand) => {
      if (cand.cnn_score < filters.minCnnScore) return false;
      if (normalizedField.length > 0 && !cand.field_id.toLowerCase().includes(normalizedField)) {
        return false;
      }
      if (filters.dispositionFilter !== "all") {
        const disp = dispositions[cand.candidate_id] ?? "unreviewed";
        if (disp !== filters.dispositionFilter) return false;
      }
      return true;
    });

    return [...filtered].sort((a, b) => {
      const valA = a[filters.sortBy];
      const valB = b[filters.sortBy];
      const diff = valA - valB;
      return filters.sortDirection === "asc" ? diff : -diff;
    });
  }, [allCandidates, filters, dispositions]);

  const handleSetDisposition = useCallback((candidateId: string, disposition: TriageDisposition) => {
    setDispositions((prev) => ({
      ...prev,
      [candidateId]: disposition,
    }));
  }, []);

  // Synchronize initial settings from backend
  useEffect(() => {
    let isMounted = true;
    async function fetchBackendSettings(): Promise<void> {
      try {
        const resp = await requestJson<{ archive_storage_path: string }>("/api/v1/settings");
        if (isMounted && resp?.archive_storage_path) {
          setSettings((prev) => ({
            ...prev,
            archive_storage_path: resp.archive_storage_path,
          }));
        }
      } catch {
        // Backend unavailable
      }
    }
    void fetchBackendSettings();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSaveSettings = (newSettings: GlobalProgramSettings): void => {
    setSettings(newSettings);
    if (newSettings.backend_connection === "remote" && newSettings.remote_base_url) {
      setApiBaseUrl(newSettings.remote_base_url);
    } else {
      setApiBaseUrl("");
    }
    try {
      localStorage.setItem("lumia.ui.settings.v1", JSON.stringify(newSettings));
    } catch {
      // Storage unavailable
    }
    try {
      void requestJson<{ archive_storage_path: string }>("/api/v1/settings", {
        method: "POST",
        body: JSON.stringify({
          archive_storage_path: newSettings.archive_storage_path,
        }),
      });
    } catch {
      // Offline / backend unavailable
    }
    void loadDashboardTelemetry();
  };

  const handleReconnectWs = useCallback(() => {
    if (wsClientRef.current) {
      void wsClientRef.current.resetAndReconnect();
    }
  }, []);

  const handleResetDuckDb = useCallback(() => {
    void loadDashboardTelemetry();
  }, [loadDashboardTelemetry]);

  const handleSubmitPipelineJob = async (req: JobSubmissionRequest): Promise<void> => {
    if (connectionMode === "offline_simulation") {
      // Simulate successful job submission
      const newJob: WsJobSummary = {
        job_id: req.job_id,
        target_field_id: req.target_field,
        observation_night_utc: req.observation_night_utc ?? new Date().toISOString().slice(0, 10),
        status: "queued",
        matched_catalog_sources: null,
        brightest_catalog_mag_g: null,
        completed_tasks: 0,
        total_tasks: 5,
        submitted_at_utc: new Date().toISOString(),
        started_at_utc: null,
        completed_at_utc: null,
      };
      setActiveJobs((prev) => [newJob, ...prev]);
      return;
    }

    await requestJson<void>("/api/v1/jobs", {
      method: "POST",
      body: JSON.stringify(req),
    });
  };

  return (
    <ObservatoryAppShell
      status={status}
      summary={summary}
      connectionMode={connectionMode}
      wsStatus={wsStatus}
      isRefreshing={isRefreshing}
      lastUpdatedIso={lastUpdatedIso}
      activeView={activeView}
      triageQueueCount={filteredCandidates.length}
      authEnabled={settings.auth_enabled}
      operatorUsername="obs_operator_01"
      onRefresh={() => void loadDashboardTelemetry()}
      onReconnectWs={handleReconnectWs}
      onResetDuckDb={handleResetDuckDb}
      onSelectView={setActiveView}
    >
      {connectionNotice &&
        connectionMode === "offline_simulation" &&
        !(status?.status === "operational" || status?.duckdb_connected) && (
          <div
            role="alert"
            className="mb-3 rounded border border-amber-800/60 bg-amber-950/30 px-4 py-2 text-xs text-amber-200"
          >
            {connectionNotice}
          </div>
        )}

      {activeView === "triage" && (
        <CandidateTriageWorkspace
          candidates={filteredCandidates}
          selectedCandidateId={selectedCandidateId}
          dispositions={dispositions}
          filters={filters}
          isOffline={connectionMode === "offline_simulation"}
          onSelectCandidate={setSelectedCandidateId}
          onFilterChange={setFilters}
          onSetDisposition={handleSetDisposition}
        />
      )}

      {activeView === "pipeline" && (
        <PipelineDashboardView
          wsStatus={wsStatus}
          activeJobs={activeJobs}
          tasks={tasks}
          logs={logs}
          latestMetrics={latestMetrics}
          droppedLogFrames={droppedLogFrames}
          onSubmitJob={handleSubmitPipelineJob}
        />
      )}

      {activeView === "settings" && (
        <GlobalOptionsView
          settings={settings}
          onSaveSettings={handleSaveSettings}
        />
      )}
    </ObservatoryAppShell>
  );
};

export default App;

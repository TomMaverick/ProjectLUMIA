/**
 * Strict TypeScript contracts matching the FastAPI backend Pydantic schemas
 * defined in src/LUMIA/api/schemas.py and docs/modules/module_10_api.md.
 */

export type BackendOperationalState = "operational" | "degraded";

export type ConnectionMode = "connected" | "offline_simulation";

export interface PipelineStatusResponse {
  readonly status: BackendOperationalState;
  readonly duckdb_connected: boolean;
  readonly postgres_connected?: boolean;
  readonly total_jobs: number;
  readonly theme_mode: "dark";
  readonly accent_color_hex: string;
}

export interface DetectionSummaryResponse {
  readonly total_candidates: number;
  readonly high_confidence_count: number;
  readonly distinct_fields: number;
  readonly mean_snr: number | null;
  readonly brightest_mag: number | null;
}

export interface CandidateDetection {
  readonly candidate_id: string;
  readonly field_id: string;
  readonly ra_deg: number;
  readonly dec_deg: number;
  readonly bjd_tdb: number;
  readonly mag: number;
  readonly mag_err?: number;
  readonly snr: number;
  readonly cnn_score: number;
  readonly model_version?: string | null;
  readonly astrometric_residual_arcsec: number;
  readonly astrometric_uncertainty_arcsec?: number;
  readonly tracklet_id?: string | null;
  readonly survey_id?: string;
  readonly filter_band?: string;
  readonly observer_pos_au?: readonly [number, number, number] | null;
  readonly observer_vel_au_per_day?: readonly [number, number, number] | null;
  readonly light_travel_time_sec?: number | null;
}

export interface CandidateListResponse {
  readonly total_returned: number;
  readonly candidates: readonly CandidateDetection[];
}

export interface CandidateCutoutEpoch {
  readonly epoch_index: number;
  readonly bjd_tdb: number;
  readonly mag: number;
  readonly mag_err: number;
  readonly target_x_px: number;
  readonly target_y_px: number;
  readonly pixels: readonly (readonly number[])[];
  readonly reference_pixels?: readonly (readonly number[])[] | null;
  readonly difference_pixels?: readonly (readonly number[])[] | null;
  readonly survey_id?: string;
  readonly filter_band?: string;
  readonly pixel_scale_arcsec?: number;
  readonly fov_arcsec?: number | null;
}

export interface CandidateCutoutSequenceResponse {
  readonly candidate_id: string;
  readonly field_id: string;
  readonly cutout_size: number;
  readonly epochs: readonly CandidateCutoutEpoch[];
  readonly survey_id?: string;
}

export type TriageDisposition =
  | "unreviewed"
  | "confirmed_real"
  | "follow_up"
  | "flagged_bogus";

export interface CandidateFilterParams {
  readonly minCnnScore: number;
  readonly fieldId: string;
  readonly dispositionFilter: "all" | TriageDisposition;
  readonly sortBy: "cnn_score" | "snr" | "mag" | "astrometric_residual_arcsec";
  readonly sortDirection: "asc" | "desc";
}

export interface CandidateTriageRequest {
  readonly disposition: TriageDisposition;
  readonly notes?: string | null;
}

export interface CandidateTriageResponse {
  readonly triage_event_id: number;
  readonly candidate_id: string;
  readonly job_id: string;
  readonly field_id: string;
  readonly previous_disposition: TriageDisposition;
  readonly disposition: TriageDisposition;
  readonly reviewed_by_operator_id: string;
  readonly cnn_score_at_review: number;
  readonly model_version_at_review: string | null;
  readonly queued_for_hdf5_export: boolean;
  readonly decided_at_utc: string;
}

export type StretchMode = "zscale" | "asinh" | "sqrt" | "linear";

export type ColorMapMode = "grayscale" | "violet" | "inverted" | "zogy_diverging";

export type CutoutChannel = "science" | "reference" | "difference";

// Authentication Types
export type OperatorRole = "admin" | "astronomer" | "reviewer" | "service";

export interface OperatorProfile {
  readonly operator_id: string;
  readonly username: string;
  readonly email: string;
  readonly role: OperatorRole;
}

export interface AccessTokenResponse {
  readonly access_token: string;
  readonly token_type: "bearer";
  readonly expires_in: number;
  readonly operator: OperatorProfile;
}

export interface WebSocketTicketResponse {
  readonly ticket: string;
  readonly expires_in: number;
}

// WebSocket Job Streaming Types (/api/v1/ws/jobs)
export interface WsJobSummary {
  readonly job_id: string;
  readonly target_field_id: string;
  readonly observation_night_utc: string;
  readonly status: "queued" | "running" | "completed" | "failed" | "cancelled";
  readonly matched_catalog_sources: number | null;
  readonly brightest_catalog_mag_g: number | null;
  readonly completed_tasks: number;
  readonly total_tasks: number;
  readonly submitted_at_utc: string;
  readonly started_at_utc: string | null;
  readonly completed_at_utc: string | null;
}

export interface WsSnapshotPayload {
  readonly subscribed_job_id: string | null;
  readonly active_jobs: readonly WsJobSummary[];
}

export interface WsJobStateChangedPayload {
  readonly job_id: string;
  readonly target_field_id: string;
  readonly previous_status: string;
  readonly status: "queued" | "running" | "completed" | "failed" | "cancelled";
  readonly prefect_flow_run_id: string | null;
  readonly matched_catalog_sources: number | null;
  readonly parquet_partition_uri: string | null;
  readonly error_summary: string | null;
  readonly started_at_utc: string | null;
  readonly completed_at_utc: string | null;
}

export interface WsTaskStateChangedPayload {
  readonly task_run_id: string;
  readonly job_id: string;
  readonly task_name: string;
  readonly module_number: number;
  readonly status: "pending" | "running" | "retrying" | "completed" | "failed" | "skipped";
  readonly attempt_count: number;
  readonly max_retries: number;
  readonly next_retry_at_utc: string | null;
  readonly started_at_utc: string | null;
  readonly finished_at_utc: string | null;
}

export interface WsJobMetricsUpdatedPayload {
  readonly job_id: string;
  readonly field_id: string;
  readonly extracted_candidates: number;
  readonly cnn_passed_candidates: number;
  readonly linked_tracklets: number;
  readonly brightest_candidate_mag: number | null;
  readonly mean_candidate_snr: number | null;
}

export interface WsTaskLogAppendedPayload {
  readonly log_id: number;
  readonly job_id: string;
  readonly task_run_id: string;
  readonly attempt_number: number;
  readonly severity: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  readonly event_code: string | null;
  readonly message: string;
  readonly logged_at_utc: string;
}

export interface WsHeartbeatPayload {
  readonly server_status: string;
  readonly dropped_log_frames: number;
}

export interface WsErrorPayload {
  readonly error_code: string;
  readonly message: string;
}

export type WsEventType =
  | "snapshot"
  | "job.state_changed"
  | "task.state_changed"
  | "job.metrics_updated"
  | "task.log_appended"
  | "heartbeat"
  | "error";

export interface WsEnvelope<T = unknown> {
  readonly schema_version: "1.0";
  readonly event_id: string;
  readonly event_type: WsEventType;
  readonly sequence_no: number;
  readonly emitted_at_utc: string;
  readonly payload: T;
}

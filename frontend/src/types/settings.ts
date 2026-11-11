export interface DuckDbConfigSettings {
  readonly engine_mode: "in_memory_parquet" | "persistent_file";
  readonly database_file_path: string;
  readonly parquet_root_dir: string;
  readonly read_only_ui_readers: boolean;
  readonly memory_limit_gb: number;
  readonly worker_threads: number;
}

export interface PostgresConfigSettings {
  readonly connection_source: "structured_params" | "env_secret_ref";
  readonly host: string;
  readonly port: number;
  readonly database: string;
  readonly username: string;
  readonly password_secret_input?: string;
  readonly has_password_configured: boolean;
  readonly env_password_var: string | null;
  readonly ssl_mode: "disable" | "prefer" | "require" | "verify-ca" | "verify-full";
  readonly pool_size: number;
  readonly max_overflow: number;
}

export interface DisplayErgonomicsSettings {
  readonly ui_scale_percent: 100 | 110 | 120 | 130 | 150;
  readonly default_cadence_hz: number;
  readonly default_stretch_mode: "zscale" | "asinh" | "sqrt" | "linear";
  readonly default_colormap: "violet" | "grayscale" | "inverted";
  readonly default_diff_colormap: "zogy_diverging" | "violet" | "grayscale";
  readonly auto_advance_on_triage: boolean;
  readonly log_terminal_buffer_size: number;
}

export interface GlobalProgramSettings {
  readonly backend_connection: "local" | "remote";
  readonly remote_base_url: string;
  readonly auth_enabled: boolean;
  readonly archive_storage_path: string;
  readonly duckdb: DuckDbConfigSettings;
  readonly postgres: PostgresConfigSettings;
  readonly display: DisplayErgonomicsSettings;
}

export interface CatalogQueryParams {
  readonly ra_min_deg: number;
  readonly ra_max_deg: number;
  readonly dec_min_deg: number;
  readonly dec_max_deg: number;
  readonly max_mag: number;
}

export interface JobSubmissionRequest {
  readonly job_id: string;
  readonly target_field: string;
  readonly observation_night_utc?: string;
  readonly profile_id?: string | null;
  readonly query_params: CatalogQueryParams;
}

export interface JobStatusResponse {
  readonly job_id: string;
  readonly target_field_id: string;
  readonly observation_night_utc: string;
  readonly profile_id?: string | null;
  readonly status: "queued" | "running" | "completed" | "failed" | "cancelled";
  readonly matched_catalog_sources: number | null;
  readonly brightest_catalog_mag_g: number | null;
  readonly completed_tasks?: number;
  readonly total_tasks?: number;
  readonly parquet_partition_uri?: string | null;
  readonly error_summary?: string | null;
  readonly submitted_at_utc: string;
  readonly started_at_utc: string | null;
  readonly completed_at_utc: string | null;
}

import type {
  CandidateCutoutEpoch,
  CandidateCutoutSequenceResponse,
  CandidateDetection,
  CandidateListResponse,
  CandidateTriageRequest,
  CandidateTriageResponse,
  DetectionSummaryResponse,
  PipelineStatusResponse,
} from "../types/api";
import { requestJson } from "./httpClient";

export async function fetchPipelineStatus(
  signal?: AbortSignal,
): Promise<PipelineStatusResponse> {
  return requestJson<PipelineStatusResponse>("/api/v1/status", { signal });
}

export async function fetchDetectionSummary(
  minCnnScore = 0.8,
  signal?: AbortSignal,
): Promise<DetectionSummaryResponse> {
  const params = new URLSearchParams({
    min_cnn_score: minCnnScore.toFixed(2),
  });
  return requestJson<DetectionSummaryResponse>(
    `/api/v1/detections/summary?${params.toString()}`,
    { signal },
  );
}

export interface FetchCandidatesOptions {
  readonly minCnnScore?: number;
  readonly fieldId?: string;
  readonly limit?: number;
  readonly offset?: number;
  readonly signal?: AbortSignal;
}

export async function fetchCandidates(
  options: FetchCandidatesOptions = {},
): Promise<CandidateListResponse> {
  const params = new URLSearchParams();
  if (options.minCnnScore !== undefined) {
    params.set("min_cnn_score", options.minCnnScore.toFixed(2));
  }
  if (options.fieldId && options.fieldId.trim().length > 0) {
    params.set("field_id", options.fieldId.trim());
  }
  params.set("limit", String(options.limit ?? 100));
  params.set("offset", String(options.offset ?? 0));

  return requestJson<CandidateListResponse>(
    `/api/v1/candidates?${params.toString()}`,
    { signal: options.signal },
  );
}

export async function fetchCandidateCutouts(
  candidateId: string,
  signal?: AbortSignal,
): Promise<CandidateCutoutSequenceResponse> {
  const encodedId = encodeURIComponent(candidateId);
  return requestJson<CandidateCutoutSequenceResponse>(
    `/api/v1/candidates/${encodedId}/cutouts`,
    { signal },
  );
}

export async function submitCandidateTriage(
  candidateId: string,
  body: CandidateTriageRequest,
  signal?: AbortSignal,
): Promise<CandidateTriageResponse> {
  const encodedId = encodeURIComponent(candidateId);
  return requestJson<CandidateTriageResponse>(
    `/api/v1/candidates/${encodedId}/triage`,
    {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    },
  );
}

// ---------------------------------------------------------------------------
// Deterministic Offline Simulation Dataset (used only when backend is offline)
// ---------------------------------------------------------------------------

export const OFFLINE_SIMULATION_CANDIDATES: readonly CandidateDetection[] = [
  {
    candidate_id: "sim-neo-001",
    field_id: "LUMIA_F0412",
    ra_deg: 184.6125,
    dec_deg: 14.882,
    bjd_tdb: 2460755.74125,
    mag: 20.18,
    mag_err: 0.08,
    snr: 15.4,
    cnn_score: 0.97,
    model_version: "resnet_lite_s31_v2",
    astrometric_residual_arcsec: 0.09,
    tracklet_id: "trk-0412-01",
  },
  {
    candidate_id: "sim-mba-002",
    field_id: "LUMIA_F0412",
    ra_deg: 184.9042,
    dec_deg: 15.106,
    bjd_tdb: 2460755.74125,
    mag: 21.05,
    mag_err: 0.12,
    snr: 9.8,
    cnn_score: 0.88,
    model_version: "resnet_lite_s31_v2",
    astrometric_residual_arcsec: 0.14,
    tracklet_id: "trk-0412-02",
  },
  {
    candidate_id: "sim-tno-003",
    field_id: "LUMIA_F0419",
    ra_deg: 186.118,
    dec_deg: 12.451,
    bjd_tdb: 2460755.7621,
    mag: 21.84,
    mag_err: 0.18,
    snr: 6.9,
    cnn_score: 0.74,
    model_version: "resnet_lite_s31_v2",
    astrometric_residual_arcsec: 0.19,
    tracklet_id: null,
  },
  {
    candidate_id: "sim-art-004",
    field_id: "LUMIA_F0419",
    ra_deg: 186.395,
    dec_deg: 12.602,
    bjd_tdb: 2460755.7621,
    mag: 22.31,
    mag_err: 0.24,
    snr: 4.8,
    cnn_score: 0.28,
    model_version: "resnet_lite_s31_v2",
    astrometric_residual_arcsec: 0.42,
    tracklet_id: null,
  },
];

export const OFFLINE_SIMULATION_STATUS: PipelineStatusResponse = {
  status: "degraded",
  duckdb_connected: false,
  postgres_connected: false,
  total_jobs: 0,
  theme_mode: "dark",
  accent_color_hex: "#581c87",
};

export function buildOfflineSummary(
  candidates: readonly CandidateDetection[],
  minCnnScore = 0.8,
): DetectionSummaryResponse {
  if (candidates.length === 0) {
    return {
      total_candidates: 0,
      high_confidence_count: 0,
      distinct_fields: 0,
      mean_snr: null,
      brightest_mag: null,
    };
  }
  const highConf = candidates.filter((c) => c.cnn_score >= minCnnScore).length;
  const fields = new Set(candidates.map((c) => c.field_id)).size;
  const sumSnr = candidates.reduce((acc, c) => acc + c.snr, 0);
  const brightest = Math.min(...candidates.map((c) => c.mag));

  return {
    total_candidates: candidates.length,
    high_confidence_count: highConf,
    distinct_fields: fields,
    mean_snr: sumSnr / candidates.length,
    brightest_mag: brightest,
  };
}

function hashSeed(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function buildOfflineCutoutSequence(
  candidate: CandidateDetection,
): CandidateCutoutSequenceResponse {
  const cutoutSize = 31;
  const seedBase = hashSeed(candidate.candidate_id);
  const epochs: CandidateCutoutEpoch[] = [];

  for (let epochIdx = 0; epochIdx < 4; epochIdx += 1) {
    const targetX = 9.5 + epochIdx * 3.8;
    const targetY = 20.5 - epochIdx * 3.2;
    const bjd = candidate.bjd_tdb + epochIdx * 0.01042;
    const magDelta = (epochIdx % 2 === 0 ? -0.03 : 0.04) * (1 + epochIdx * 0.1);
    const magErr = Math.max(0.03, Number((1.0857 / Math.max(candidate.snr, 2)).toFixed(3)));

    const sciencePixels: number[][] = [];
    const referencePixels: number[][] = [];
    const differencePixels: number[][] = [];

    for (let y = 0; y < cutoutSize; y += 1) {
      const sciRow: number[] = [];
      const refRow: number[] = [];
      const diffRow: number[] = [];

      for (let x = 0; x < cutoutSize; x += 1) {
        const cellHash = hashSeed(`${seedBase}-${epochIdx}-${x}-${y}`);
        const u1 = ((cellHash & 0xffff) + 1) / 65537.0;
        const u2 = (((cellHash >>> 16) & 0xffff) + 1) / 65537.0;
        const noise =
          Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * 0.8;

        // Static reference star
        const staticStar =
          18.0 * Math.exp(-((x - 7.0) ** 2 + (y - 8.0) ** 2) / 3.2);

        // Moving candidate PSF
        const distSq = (x - targetX) ** 2 + (y - targetY) ** 2;
        const psfPeak = candidate.snr * 1.5;
        const movingPsf = psfPeak * Math.exp(-distSq / (2.0 * 1.35 ** 2));

        const sciVal = Number((10.0 + noise + staticStar + movingPsf).toFixed(3));
        const refVal = Number((10.0 + noise * 0.5 + staticStar).toFixed(3));
        const diffVal = Number((noise + movingPsf).toFixed(3));

        sciRow.push(sciVal);
        refRow.push(refVal);
        diffRow.push(diffVal);
      }
      sciencePixels.push(sciRow);
      referencePixels.push(refRow);
      differencePixels.push(diffRow);
    }

    epochs.push({
      epoch_index: epochIdx,
      bjd_tdb: Number(bjd.toFixed(5)),
      mag: Number((candidate.mag + magDelta).toFixed(2)),
      mag_err: magErr,
      target_x_px: Number(targetX.toFixed(2)),
      target_y_px: Number(targetY.toFixed(2)),
      pixels: sciencePixels,
      reference_pixels: referencePixels,
      difference_pixels: differencePixels,
    });
  }

  return {
    candidate_id: candidate.candidate_id,
    field_id: candidate.field_id,
    cutout_size: cutoutSize,
    epochs,
  };
}

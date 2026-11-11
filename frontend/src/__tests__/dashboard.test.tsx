import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";

function buildMockCutouts(candidateId: string) {
  const pixels: number[][] = Array.from({ length: 32 }, () =>
    Array.from({ length: 32 }, () => 1.5),
  );
  return {
    candidate_id: candidateId,
    field_id: "ZTF_0412",
    cutout_size: 32,
    epochs: [
      {
        epoch_index: 0,
        bjd_tdb: 2460755.75658,
        mag: 20.41,
        mag_err: 0.08,
        target_x_px: 12.2,
        target_y_px: 18.4,
        pixels,
      },
      {
        epoch_index: 1,
        bjd_tdb: 2460755.767,
        mag: 20.38,
        mag_err: 0.08,
        target_x_px: 14.4,
        target_y_px: 16.8,
        pixels,
      },
      {
        epoch_index: 2,
        bjd_tdb: 2460755.77742,
        mag: 20.44,
        mag_err: 0.08,
        target_x_px: 16.6,
        target_y_px: 15.2,
        pixels,
      },
      {
        epoch_index: 3,
        bjd_tdb: 2460755.78784,
        mag: 20.39,
        mag_err: 0.08,
        target_x_px: 18.8,
        target_y_px: 13.6,
        pixels,
      },
    ],
  };
}

function installFetchMock() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = typeof input === "string" ? input : input.toString();

    if (url.includes("/api/v1/status")) {
      return new Response(
        JSON.stringify({
          status: "operational",
          duckdb_connected: true,
          total_jobs: 3,
          theme_mode: "dark",
          accent_color_hex: "#581c87",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }

    if (url.includes("/api/v1/detections/summary")) {
      return new Response(
        JSON.stringify({
          total_candidates: 2,
          high_confidence_count: 1,
          distinct_fields: 1,
          mean_snr: 10.0,
          brightest_mag: 20.4,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }

    if (url.includes("/cutouts")) {
      const id = url.includes("cand-002") ? "cand-002" : "cand-001";
      return new Response(JSON.stringify(buildMockCutouts(id)), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.includes("/api/v1/candidates")) {
      return new Response(
        JSON.stringify({
          total_returned: 2,
          candidates: [
            {
              candidate_id: "cand-001",
              field_id: "ZTF_0412",
              ra_deg: 180.125,
              dec_deg: 15.34,
              bjd_tdb: 2460755.75658,
              mag: 20.4,
              snr: 12.5,
              cnn_score: 0.96,
              astrometric_residual_arcsec: 0.11,
            },
            {
              candidate_id: "cand-002",
              field_id: "ZTF_0412",
              ra_deg: 180.44,
              dec_deg: 15.61,
              bjd_tdb: 2460755.75662,
              mag: 21.2,
              snr: 7.5,
              cnn_score: 0.62,
              astrometric_residual_arcsec: 0.21,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }

    if (url.includes("/api/v1/settings")) {
      return new Response(
        JSON.stringify({
          archive_storage_path: "data/archives/",
          num_cpu_threads: 4,
          max_gpu_memory_gb: 4.0,
          enable_openmp_threading_control: true,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response("Not Found", { status: 404 });
  });
}

import {
  formatDecSexagesimal,
  formatRaSexagesimal,
  normalizeCutoutPixels,
} from "../utils/astronomyFormat";

describe("LUMIA Observatory Console Dashboard", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("connects to FastAPI endpoints and displays pipeline health, blink-loop, and candidate triage", async () => {
    installFetchMock();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/OPERATIONAL/i)).toBeInTheDocument();
    });

    expect(screen.getAllByText("cand-001").length).toBeGreaterThan(0);
    expect(screen.getAllByText("cand-002").length).toBeGreaterThan(0);
    expect(screen.getByText(/0\.11"/)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Epoch 1 \/ 4/i)).toBeInTheDocument();
    });

    // Pause automatic blink-loop and step manually to Epoch 2
    const pauseBtn = screen.getByRole("button", { name: /pause blink loop/i });
    fireEvent.click(pauseBtn);

    const nextBtn = screen.getByRole("button", { name: /next epoch/i });
    fireEvent.click(nextBtn);

    await waitFor(() => {
      expect(screen.getByText(/Epoch 2 \/ 4/i)).toBeInTheDocument();
    });

    // Select second candidate in triage table and verify detail panel updates
    const secondRow = screen.getByRole("row", { name: /cand-002/i });
    fireEvent.click(secondRow);

    await waitFor(() => {
      expect(screen.getByText(/62\.0%/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/0\.21"/)).toBeInTheDocument();
  });

  it("filters candidates by minimum CNN score and updates operator triage dispositions", async () => {
    installFetchMock();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/OPERATIONAL/i)).toBeInTheDocument();
    });

    // Confirm cand-001 as Real and verify disposition badge updates in table
    const confirmRealBtn = screen.getByRole("button", {
      name: /confirm real/i,
    });
    fireEvent.click(confirmRealBtn);
    expect(screen.getByText("REAL")).toBeInTheDocument();

    // Raise minimum CNN score threshold to 0.80 so cand-002 (0.62) is filtered out
    const cnnSlider = screen.getByRole("slider", {
      name: /minimum cnn score filter/i,
    });
    fireEvent.change(cnnSlider, { target: { value: "0.8" } });

    expect(screen.queryByRole("row", { name: /cand-002/i })).not.toBeInTheDocument();
    expect(screen.getByRole("row", { name: /cand-001/i })).toBeInTheDocument();
  });

  it("falls back cleanly to offline preview dataset when FastAPI backend is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("Connection refused"),
    );
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/OFFLINE SIMULATION/i)).toBeInTheDocument();
    });

    expect(screen.getByRole("heading", { name: /LUMIA/i })).toBeInTheDocument();
    expect(screen.getAllByText("sim-neo-001").length).toBeGreaterThan(0);
  });

  it("formats sexagesimal ICRS coordinates and normalizes cutout pixel intensities accurately", () => {
    expect(formatRaSexagesimal(180.0)).toBe("12:00:00.00");
    expect(formatDecSexagesimal(-15.5)).toBe("-15:30:00.0");

    const normalized = normalizeCutoutPixels(
      [
        [10.0, 20.0],
        [30.0, 50.0],
      ],
      "linear",
    );
    expect(normalized[0]?.[0]).toBeCloseTo(0.0, 5);
    expect(normalized[1]?.[1]).toBeCloseTo(1.0, 5);
  });

  it("navigates fluently between Triage, Pipeline, and Global Settings views", async () => {
    installFetchMock();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/OPERATIONAL/i)).toBeInTheDocument();
    });

    // Default view: Candidate Triage
    expect(screen.getByRole("heading", { name: /multi-epoch 3-channel blink matrix/i })).toBeInTheDocument();

    // Switch to Live Pipeline view
    const pipelineNavBtn = screen.getByRole("button", { name: /live pipeline/i });
    fireEvent.click(pipelineNavBtn);

    expect(screen.getByRole("heading", { name: /live pipeline monitoring/i })).toBeInTheDocument();
    expect(screen.getByText(/submit pipeline job/i)).toBeInTheDocument();

    // Switch to Global Settings view
    const settingsNavBtn = screen.getByRole("button", { name: /global settings/i });
    fireEvent.click(settingsNavBtn);

    expect(screen.getByRole("heading", { name: /global program configuration/i })).toBeInTheDocument();
    expect(screen.getByText(/duckdb analytical engine/i)).toBeInTheDocument();
    expect(screen.getByText(/postgresql transactional engine/i)).toBeInTheDocument();

    // Switch back to Candidate Triage view
    const triageNavBtn = screen.getByRole("button", { name: /candidate triage/i });
    fireEvent.click(triageNavBtn);

    expect(screen.getByRole("heading", { name: /multi-epoch 3-channel blink matrix/i })).toBeInTheDocument();
  });

  it("displays full subtitle, local time, and supports remote server backend connection configuration", async () => {
    installFetchMock();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/OPERATIONAL/i)).toBeInTheDocument();
    });

    // Check header subtitle and local clock
    expect(
      screen.getByText("Low-SNR Unidentified Moving Interplanetary Analyzer"),
    ).toBeInTheDocument();
    expect(screen.getByText(/LOCAL \d{2}:\d{2}/)).toBeInTheDocument();

    // Verify clickable status pills
    const wsBtn = screen.getByRole("button", { name: /web socket:/i });
    expect(wsBtn).toBeInTheDocument();
    fireEvent.click(wsBtn);

    const duckdbBtn = screen.getByRole("button", { name: /duckdb:/i });
    expect(duckdbBtn).toBeInTheDocument();
    fireEvent.click(duckdbBtn);

    // Navigate to Global Settings and inspect Backend Connection
    const settingsNavBtn = screen.getByRole("button", { name: /global settings/i });
    fireEvent.click(settingsNavBtn);

    const connectionTabBtn = screen.getByRole("button", { name: /backend connection/i });
    fireEvent.click(connectionTabBtn);

    expect(screen.getByText(/Backend Connection & Target Host/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Local Device \(Standard\)/i)).toBeInTheDocument();

    const remoteRadio = screen.getByLabelText(/Remote Server/i);
    expect(remoteRadio).toBeInTheDocument();
    fireEvent.click(remoteRadio);

    const urlInput = screen.getByPlaceholderText("http://192.168.1.100:8000");
    expect(urlInput).toBeInTheDocument();
    fireEvent.change(urlInput, { target: { value: "http://192.168.1.50:8000" } });

    // Save settings
    const saveBtn = screen.getByRole("button", { name: /save & apply/i });
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(screen.getByText(/Settings Applied/i)).toBeInTheDocument();
    });
  });
});


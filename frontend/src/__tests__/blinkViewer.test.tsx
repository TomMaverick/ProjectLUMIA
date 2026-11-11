import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  BlinkLoopViewer,
  getSurveyBadgeDetails,
} from "../components/BlinkLoopViewer";
import type { CandidateCutoutSequenceResponse } from "../types/api";

function buildMultiSurveyCutouts(): CandidateCutoutSequenceResponse {
  const pixels32 = Array.from({ length: 32 }, () =>
    Array.from({ length: 32 }, () => 2.0),
  );

  return {
    candidate_id: "cand-ms-001",
    field_id: "COSMOS_F01",
    cutout_size: 32,
    survey_id: "Pan-STARRS",
    epochs: [
      {
        epoch_index: 0,
        bjd_tdb: 2460755.7001,
        mag: 20.45,
        mag_err: 0.07,
        target_x_px: 15.5,
        target_y_px: 15.5,
        pixels: pixels32,
        survey_id: "ZTF",
        filter_band: "r",
        pixel_scale_arcsec: 1.012,
        fov_arcsec: 32.384,
      },
      {
        epoch_index: 1,
        bjd_tdb: 2460755.7102,
        mag: 20.5,
        mag_err: 0.08,
        target_x_px: 16.5,
        target_y_px: 15.0,
        pixels: pixels32,
        survey_id: "Pan-STARRS",
        filter_band: "g",
        pixel_scale_arcsec: 0.25,
        fov_arcsec: 8.0,
      },
      {
        epoch_index: 2,
        bjd_tdb: 2460755.7203,
        mag: 20.48,
        mag_err: 0.06,
        target_x_px: 17.5,
        target_y_px: 14.5,
        pixels: pixels32,
        survey_id: "Rubin",
        filter_band: "i",
        pixel_scale_arcsec: 0.2,
        fov_arcsec: 6.4,
      },
      {
        epoch_index: 3,
        bjd_tdb: 2460755.7304,
        mag: 20.42,
        mag_err: 0.05,
        target_x_px: 18.5,
        target_y_px: 14.0,
        pixels: pixels32,
        survey_id: "JWST",
        filter_band: "F090W",
        pixel_scale_arcsec: 0.031,
        fov_arcsec: 0.992,
      },
    ],
  };
}

describe("Interactive Multi-Survey Blink Viewer (Module 10)", () => {
  it("computes survey badges and CSS classes for diverse survey platforms", () => {
    expect(getSurveyBadgeDetails("ZTF").label).toBe("ZTF");
    expect(getSurveyBadgeDetails("Pan-STARRS").shortName).toBe("PS1");
    expect(getSurveyBadgeDetails("Rubin").shortName).toBe("LSST");
    expect(getSurveyBadgeDetails("JWST").shortName).toBe("JWST");
    expect(getSurveyBadgeDetails("Euclid").shortName).toBe("SPACE");
  });

  it("renders multi-survey fusion badge when sequence contains cutouts from disparate surveys", () => {
    const sequence = buildMultiSurveyCutouts();
    const handleEpochChange = vi.fn();

    render(
      <BlinkLoopViewer
        cutoutSequence={sequence}
        isLoading={false}
        activeEpochIndex={0}
        onEpochChange={handleEpochChange}
      />,
    );

    expect(screen.getByText(/Multi-Survey Fusion/i)).toBeInTheDocument();
    expect(screen.getByText(/Target cand-ms-001/i)).toBeInTheDocument();
  });

  it("displays active survey badge, optical bandpass tag, and pixel scale", () => {
    const sequence = buildMultiSurveyCutouts();
    const handleEpochChange = vi.fn();

    const { rerender } = render(
      <BlinkLoopViewer
        cutoutSequence={sequence}
        isLoading={false}
        activeEpochIndex={0}
        onEpochChange={handleEpochChange}
      />,
    );

    // Epoch 0 is ZTF (r-band)
    expect(screen.getAllByText(/ZTF/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/band: r/i)).toBeInTheDocument();
    expect(screen.getByText(/1.012"\/px/i)).toBeInTheDocument();

    // Switch to Epoch 1: Pan-STARRS (g-band, 0.250"/px)
    rerender(
      <BlinkLoopViewer
        cutoutSequence={sequence}
        isLoading={false}
        activeEpochIndex={1}
        onEpochChange={handleEpochChange}
      />,
    );

    expect(screen.getAllByText(/Pan-STARRS/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/band: g/i)).toBeInTheDocument();
    expect(screen.getByText(/0.250"\/px/i)).toBeInTheDocument();
    expect(screen.getByText(/FOV: 8.0"/i)).toBeInTheDocument();

    // Switch to Epoch 3: JWST (F090W, 0.031"/px)
    rerender(
      <BlinkLoopViewer
        cutoutSequence={sequence}
        isLoading={false}
        activeEpochIndex={3}
        onEpochChange={handleEpochChange}
      />,
    );

    expect(screen.getAllByText(/JWST/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/band: F090W/i)).toBeInTheDocument();
    expect(screen.getByText(/0.031"\/px/i)).toBeInTheDocument();
  });

  it("allows operator to toggle FOV Normalization and Astrometric Scale Bar", () => {
    const sequence = buildMultiSurveyCutouts();
    const handleEpochChange = vi.fn();

    render(
      <BlinkLoopViewer
        cutoutSequence={sequence}
        isLoading={false}
        activeEpochIndex={0}
        onEpochChange={handleEpochChange}
      />,
    );

    const fovCheckbox = screen.getByLabelText(/Normalize FOV/i);
    expect(fovCheckbox).toBeChecked();
    fireEvent.click(fovCheckbox);
    expect(fovCheckbox).not.toBeChecked();

    const scaleBarCheckbox = screen.getByLabelText(/Scale Bar/i);
    expect(scaleBarCheckbox).toBeChecked();
    fireEvent.click(scaleBarCheckbox);
    expect(scaleBarCheckbox).not.toBeChecked();
  });

  it("provides epoch buttons annotated with survey abbreviations", () => {
    const sequence = buildMultiSurveyCutouts();
    const handleEpochChange = vi.fn();

    render(
      <BlinkLoopViewer
        cutoutSequence={sequence}
        isLoading={false}
        activeEpochIndex={0}
        onEpochChange={handleEpochChange}
      />,
    );

    expect(screen.getByText(/E1/i)).toBeInTheDocument();
    expect(screen.getByText(/E2/i)).toBeInTheDocument();
    expect(screen.getByText(/E3/i)).toBeInTheDocument();
    expect(screen.getByText(/E4/i)).toBeInTheDocument();

    // Clicking E2 triggers onEpochChange with index 1
    const e2Btn = screen.getByTitle(/Pan-STARRS \(g\)/i);
    fireEvent.click(e2Btn);
    expect(handleEpochChange).toHaveBeenCalledWith(1);
  });
});

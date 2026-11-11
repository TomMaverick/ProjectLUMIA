import type {
  CandidateCutoutEpoch,
  ColorMapMode,
  StretchMode,
} from "../types/api";

/**
 * Convert Right Ascension in decimal degrees [0, 360) to ICRS sexagesimal HH:MM:SS.ss.
 */
export function formatRaSexagesimal(raDeg: number): string {
  const normalized = ((raDeg % 360) + 360) % 360;
  const totalHours = normalized / 15.0;
  const hours = Math.floor(totalHours);
  const remMinutes = (totalHours - hours) * 60.0;
  const minutes = Math.floor(remMinutes);
  const seconds = (remMinutes - minutes) * 60.0;

  const hh = String(hours).padStart(2, "0");
  const mm = String(minutes).padStart(2, "0");
  const ss = seconds.toFixed(2).padStart(5, "0");
  return `${hh}:${mm}:${ss}`;
}

/**
 * Convert Declination in decimal degrees [-90, +90] to ICRS sexagesimal +DD:MM:SS.s.
 */
export function formatDecSexagesimal(decDeg: number): string {
  const clamped = Math.max(-90.0, Math.min(90.0, decDeg));
  const sign = clamped >= 0 ? "+" : "-";
  const absDec = Math.abs(clamped);
  const degrees = Math.floor(absDec);
  const remArcmin = (absDec - degrees) * 60.0;
  const arcmin = Math.floor(remArcmin);
  const arcsec = (remArcmin - arcmin) * 60.0;

  const dd = String(degrees).padStart(2, "0");
  const mm = String(arcmin).padStart(2, "0");
  const ss = arcsec.toFixed(1).padStart(4, "0");
  return `${sign}${dd}:${mm}:${ss}`;
}

/**
 * Compute pixel displacement velocity in pixels per hour across cutout epochs.
 */
export function computePixelMotionRatePxPerHour(
  epochs: readonly CandidateCutoutEpoch[],
): number | null {
  if (epochs.length < 2) {
    return null;
  }
  const first = epochs[0];
  const last = epochs[epochs.length - 1];
  if (!first || !last) {
    return null;
  }
  const deltaDays = last.bjd_tdb - first.bjd_tdb;
  if (deltaDays <= 0) {
    return null;
  }
  const dx = last.target_x_px - first.target_x_px;
  const dy = last.target_y_px - first.target_y_px;
  const distPx = Math.hypot(dx, dy);
  return distPx / (deltaDays * 24.0);
}

/**
 * Compute normalized [0, 1] display intensity bounds for a 2D astronomical cutout.
 */
export function normalizeCutoutPixels(
  pixels: readonly (readonly number[])[],
  stretch: StretchMode,
): readonly (readonly number[])[] {
  const flat: number[] = [];
  for (const row of pixels) {
    for (const val of row) {
      if (Number.isFinite(val)) {
        flat.push(val);
      }
    }
  }

  if (flat.length === 0) {
    return pixels;
  }

  const sorted = [...flat].sort((a, b) => a - b);
  let low = sorted[0] ?? 0;
  let high = sorted[sorted.length - 1] ?? 1;

  if (stretch === "zscale") {
    // Robust percentile clipping approximating astronomical ZScale background/peak bounds
    const p10Index = Math.floor(sorted.length * 0.1);
    const p995Index = Math.min(
      sorted.length - 1,
      Math.floor(sorted.length * 0.995),
    );
    low = sorted[p10Index] ?? low;
    high = sorted[p995Index] ?? high;
  }

  const span = high - low > 1e-9 ? high - low : 1.0;

  return pixels.map((row) =>
    row.map((val) => {
      const clamped = Math.max(0.0, Math.min(1.0, (val - low) / span));
      if (stretch === "sqrt") {
        return Math.sqrt(clamped);
      }
      if (stretch === "asinh") {
        return Math.asinh(clamped * 3.0) / Math.asinh(3.0);
      }
      return clamped;
    }),
  );
}

/**
 * Map a normalized [0, 1] pixel intensity to an CSS rgb() color string.
 */
export function mapIntensityToRgb(
  intensity: number,
  colorMap: ColorMapMode,
): string {
  const t = Math.max(0.0, Math.min(1.0, intensity));

  if (colorMap === "inverted") {
    const v = Math.round((1.0 - t) * 245);
    return `rgb(${v}, ${v}, ${Math.min(255, v + 8)})`;
  }

  if (colorMap === "violet") {
    // Deep charcoal (#090a0f) -> dark purple (#3b185f / #581c87) -> lavender highlight
    const r = Math.round(9 + Math.pow(t, 0.85) * 230);
    const g = Math.round(10 + Math.pow(t, 1.35) * 195);
    const b = Math.round(15 + Math.pow(t, 0.7) * 240);
    return `rgb(${r}, ${g}, ${b})`;
  }

  if (colorMap === "zogy_diverging") {
    // Center at 0.5 (zero difference flux)
    // < 0.5 negative dipole: muted cyan (#0e7490)
    // > 0.5 positive transient: luminous violet-amber (#a855f7 -> #f5d0fe)
    if (t < 0.5) {
      const factor = (0.5 - t) * 2.0;
      const r = Math.round(9 + factor * 5);
      const g = Math.round(10 + factor * 106);
      const b = Math.round(15 + factor * 129);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const factor = (t - 0.5) * 2.0;
      const r = Math.round(9 + factor * 236);
      const g = Math.round(10 + factor * 198);
      const b = Math.round(15 + factor * 239);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  // Standard observatory grayscale with slight cool slate floor
  const r = Math.round(9 + t * 238);
  const g = Math.round(11 + t * 238);
  const b = Math.round(16 + t * 239);
  return `rgb(${r}, ${g}, ${b})`;
}

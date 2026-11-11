import React, { useState } from "react";
import { Plus, X, AlertCircle } from "lucide-react";
import type { JobSubmissionRequest } from "../../types/settings";

export interface JobSubmissionDrawerProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly onSubmitJob: (request: JobSubmissionRequest) => Promise<void>;
}

export const JobSubmissionDrawer: React.FC<JobSubmissionDrawerProps> = ({
  isOpen,
  onClose,
  onSubmitJob,
}) => {
  const [jobId, setJobId] = useState<string>(`job-${new Date().toISOString().slice(0, 10).replace(/-/g, "")}-01`);
  const [targetField, setTargetField] = useState<string>("ZTF_0412");
  const [raMin, setRaMin] = useState<number>(180.0);
  const [raMax, setRaMax] = useState<number>(185.0);
  const [decMin, setDecMin] = useState<number>(14.0);
  const [decMax, setDecMax] = useState<number>(16.0);
  const [maxMag, setMaxMag] = useState<number>(21.5);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!isOpen) {
    return null;
  }

  const isMeridianWrap = raMin > raMax;

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    setErrorMsg(null);

    if (!jobId.trim() || !targetField.trim()) {
      setErrorMsg("Job ID and Target Field are mandatory.");
      return;
    }

    if (decMin > decMax) {
      setErrorMsg("Declination min must be <= max.");
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmitJob({
        job_id: jobId.trim(),
        target_field: targetField.trim(),
        observation_night_utc: new Date().toISOString().slice(0, 10),
        query_params: {
          ra_min_deg: raMin,
          ra_max_deg: raMax,
          dec_min_deg: decMin,
          dec_max_deg: decMax,
          max_mag: maxMag,
        },
      });
      onClose();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Job submission failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-xs">
      <div className="flex h-full w-full max-w-md flex-col border-l border-[#3b185f] bg-[#11131c] shadow-2xl p-5 text-xs text-slate-200">
        <div className="flex items-center justify-between border-b border-[#26293d] pb-3 mb-4">
          <div className="flex items-center gap-2">
            <Plus className="h-4 w-4 text-purple-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-100">
              Submit Prefect Pipeline Job
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-slate-400 hover:bg-[#181b28] hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="flex-1 space-y-4 overflow-y-auto">
          {errorMsg && (
            <div className="rounded border border-rose-800/80 bg-rose-950/40 p-2.5 text-rose-300 flex items-start gap-2">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}

          <div>
            <label className="block text-[11px] font-semibold uppercase text-slate-300 mb-1">
              Job ID (SafeIdentifier)
            </label>
            <input
              type="text"
              required
              pattern="^[A-Za-z0-9_.-]+$"
              value={jobId}
              onChange={(e) => setJobId(e.target.value)}
              className="w-full rounded border border-[#3b185f] bg-[#090a0f] p-2 font-mono text-slate-100 focus:outline-none focus:border-[#581c87]"
            />
          </div>

          <div>
            <label className="block text-[11px] font-semibold uppercase text-slate-300 mb-1">
              Target Field ID
            </label>
            <input
              type="text"
              required
              value={targetField}
              onChange={(e) => setTargetField(e.target.value)}
              className="w-full rounded border border-[#3b185f] bg-[#090a0f] p-2 font-mono text-slate-100 focus:outline-none focus:border-[#581c87]"
            />
          </div>

          {/* Sky-Box Coordinates */}
          <div className="rounded border border-[#26293d] bg-[#090a0f] p-3 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-semibold uppercase text-[10px] text-purple-300">
                Catalog Sky-Box Filter (DuckDB)
              </span>
              {isMeridianWrap && (
                <span className="rounded bg-amber-950/70 border border-amber-600/70 px-1.5 py-0.5 text-[9px] font-mono text-amber-200">
                  0°/360° Meridian Wrap Active
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] uppercase text-slate-400 mb-0.5">RA Min (deg)</label>
                <input
                  type="number"
                  step="0.001"
                  min="0"
                  max="360"
                  value={raMin}
                  onChange={(e) => setRaMin(Number(e.target.value))}
                  className="w-full rounded border border-[#3b185f] bg-[#11131c] p-1.5 font-mono text-slate-100 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[10px] uppercase text-slate-400 mb-0.5">RA Max (deg)</label>
                <input
                  type="number"
                  step="0.001"
                  min="0"
                  max="360"
                  value={raMax}
                  onChange={(e) => setRaMax(Number(e.target.value))}
                  className="w-full rounded border border-[#3b185f] bg-[#11131c] p-1.5 font-mono text-slate-100 focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] uppercase text-slate-400 mb-0.5">Dec Min (deg)</label>
                <input
                  type="number"
                  step="0.001"
                  min="-90"
                  max="90"
                  value={decMin}
                  onChange={(e) => setDecMin(Number(e.target.value))}
                  className="w-full rounded border border-[#3b185f] bg-[#11131c] p-1.5 font-mono text-slate-100 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[10px] uppercase text-slate-400 mb-0.5">Dec Max (deg)</label>
                <input
                  type="number"
                  step="0.001"
                  min="-90"
                  max="90"
                  value={decMax}
                  onChange={(e) => setDecMax(Number(e.target.value))}
                  className="w-full rounded border border-[#3b185f] bg-[#11131c] p-1.5 font-mono text-slate-100 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-[10px] uppercase text-slate-400 mb-0.5">Max Mag Cut (mag)</label>
              <input
                type="number"
                step="0.1"
                value={maxMag}
                onChange={(e) => setMaxMag(Number(e.target.value))}
                className="w-full rounded border border-[#3b185f] bg-[#11131c] p-1.5 font-mono text-slate-100 focus:outline-none"
              />
            </div>
          </div>

          <div className="pt-4 border-t border-[#26293d] flex gap-2">
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex-1 rounded border border-[#581c87] bg-[#581c87] py-2 text-xs font-bold text-white hover:bg-[#7e22ce] transition-colors disabled:opacity-50"
            >
              {isSubmitting ? "Submitting Flow..." : "Launch Pipeline Job"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded border border-[#26293d] bg-[#181b28] px-4 py-2 text-slate-300 hover:bg-[#26293d]"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

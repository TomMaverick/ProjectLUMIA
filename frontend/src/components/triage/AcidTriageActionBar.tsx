import React, { useState } from "react";
import { Check, HelpCircle, X, RotateCcw } from "lucide-react";
import type { TriageDisposition } from "../../types/api";

export interface AcidTriageActionBarProps {
  readonly candidateId: string | null;
  readonly disposition: TriageDisposition;
  readonly isSubmitting?: boolean;
  readonly autoAdvance: boolean;
  readonly onSetDisposition: (disposition: TriageDisposition, notes?: string | null) => void;
  readonly onToggleAutoAdvance: (val: boolean) => void;
}

export const AcidTriageActionBar: React.FC<AcidTriageActionBarProps> = ({
  candidateId,
  disposition,
  isSubmitting = false,
  autoAdvance,
  onSetDisposition,
  onToggleAutoAdvance,
}) => {
  const [notes, setNotes] = useState<string>("");
  const [showNotesDrawer, setShowNotesDrawer] = useState<boolean>(false);

  const handleAction = (targetDisp: TriageDisposition): void => {
    if (!candidateId || isSubmitting) return;
    onSetDisposition(targetDisp, notes.trim().length > 0 ? notes.trim() : null);
    if (showNotesDrawer) {
      setNotes("");
      setShowNotesDrawer(false);
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-[#3b185f] bg-[#11131c] p-3 shadow-md">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
            ACID Triage Actions
          </span>
          <span className="text-[10px] text-slate-400 font-mono hidden sm:inline">
            (Keys: 1=Real, 2=Follow, 3=Bogus, 0=Reset)
          </span>
        </div>

        <div className="flex items-center gap-3">
          <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={autoAdvance}
              onChange={(e) => onToggleAutoAdvance(e.target.checked)}
              className="accent-[#581c87]"
            />
            <span>Auto-Advance</span>
          </label>

          <button
            type="button"
            onClick={() => setShowNotesDrawer((prev) => !prev)}
            className="text-[11px] text-purple-300 hover:text-purple-200 underline"
          >
            {showNotesDrawer ? "Hide Notes" : "+ Notes"}
          </button>
        </div>
      </div>

      {showNotesDrawer && (
        <div className="mt-1">
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Operator inspection notes (e.g., clean linear motion in ZOGY difference)..."
            rows={2}
            maxLength={2000}
            className="w-full rounded border border-[#3b185f] bg-[#090a0f] p-2 text-xs text-slate-200 placeholder-slate-500 focus:border-[#581c87] focus:outline-none"
          />
        </div>
      )}

      {/* Button Row */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {/* Confirm Real */}
        <button
          type="button"
          disabled={!candidateId || isSubmitting}
          onClick={() => handleAction("confirmed_real")}
          className={`flex items-center justify-center gap-1.5 rounded border py-2 text-xs font-bold transition-colors ${
            disposition === "confirmed_real"
              ? "border-emerald-500 bg-emerald-900/80 text-emerald-100 shadow-inner ring-1 ring-emerald-400"
              : "border-emerald-800/80 bg-emerald-950/40 text-emerald-200 hover:bg-emerald-900/50 hover:border-emerald-600"
          } disabled:opacity-40`}
        >
          <Check className="h-4 w-4" />
          <span>[1] Confirm Real</span>
        </button>

        {/* Follow-Up */}
        <button
          type="button"
          disabled={!candidateId || isSubmitting}
          onClick={() => handleAction("follow_up")}
          className={`flex items-center justify-center gap-1.5 rounded border py-2 text-xs font-bold transition-colors ${
            disposition === "follow_up"
              ? "border-purple-400 bg-[#581c87] text-white shadow-inner ring-1 ring-purple-300"
              : "border-[#581c87]/70 bg-[#24103a]/50 text-purple-200 hover:bg-[#3b185f]"
          } disabled:opacity-40`}
        >
          <HelpCircle className="h-4 w-4" />
          <span>[2] Follow-Up</span>
        </button>

        {/* Flag Bogus */}
        <button
          type="button"
          disabled={!candidateId || isSubmitting}
          onClick={() => handleAction("flagged_bogus")}
          className={`flex items-center justify-center gap-1.5 rounded border py-2 text-xs font-bold transition-colors ${
            disposition === "flagged_bogus"
              ? "border-rose-500 bg-rose-900/80 text-rose-100 shadow-inner ring-1 ring-rose-400"
              : "border-rose-800/80 bg-rose-950/40 text-rose-200 hover:bg-rose-900/50 hover:border-rose-600"
          } disabled:opacity-40`}
        >
          <X className="h-4 w-4" />
          <span>[3] Flag Bogus</span>
        </button>

        {/* Reset Unreviewed */}
        <button
          type="button"
          disabled={!candidateId || isSubmitting || disposition === "unreviewed"}
          onClick={() => handleAction("unreviewed")}
          className="flex items-center justify-center gap-1.5 rounded border border-[#26293d] bg-[#181b28] py-2 text-xs font-medium text-slate-300 hover:bg-[#26293d] disabled:opacity-30"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>[0] Reset</span>
        </button>
      </div>
    </div>
  );
};

import React, { useEffect, useMemo, useState } from "react";
import { ThreeChannelBlinkViewer } from "./ThreeChannelBlinkViewer";
import { AcidTriageActionBar } from "./AcidTriageActionBar";
import { CandidateDetailPanel } from "./CandidateDetailPanel";
import { CandidateTriageTable } from "../CandidateTriageTable";
import type {
  CandidateCutoutSequenceResponse,
  CandidateDetection,
  CandidateFilterParams,
  TriageDisposition,
} from "../../types/api";
import {
  buildOfflineCutoutSequence,
  fetchCandidateCutouts,
  submitCandidateTriage,
} from "../../services/lumiaApi";

export interface CandidateTriageWorkspaceProps {
  readonly candidates: readonly CandidateDetection[];
  readonly selectedCandidateId: string | null;
  readonly dispositions: Readonly<Record<string, TriageDisposition>>;
  readonly filters: CandidateFilterParams;
  readonly isOffline?: boolean;
  readonly onSelectCandidate: (candidateId: string) => void;
  readonly onFilterChange: (filters: CandidateFilterParams) => void;
  readonly onSetDisposition: (candidateId: string, disposition: TriageDisposition) => void;
}

export const CandidateTriageWorkspace: React.FC<CandidateTriageWorkspaceProps> = ({
  candidates,
  selectedCandidateId,
  dispositions,
  filters,
  isOffline = false,
  onSelectCandidate,
  onFilterChange,
  onSetDisposition,
}) => {
  const [activeEpochIndex, setActiveEpochIndex] = useState<number>(0);
  const [cutoutSequence, setCutoutSequence] = useState<CandidateCutoutSequenceResponse | null>(null);
  const [isLoadingCutouts, setIsLoadingCutouts] = useState<boolean>(false);
  const [autoAdvance, setAutoAdvance] = useState<boolean>(true);
  const [isSubmittingTriage, setIsSubmittingTriage] = useState<boolean>(false);

  // Selected candidate object
  const selectedCandidate = useMemo<CandidateDetection | null>(() => {
    if (!selectedCandidateId) {
      return candidates[0] ?? null;
    }
    return candidates.find((c) => c.candidate_id === selectedCandidateId) ?? candidates[0] ?? null;
  }, [candidates, selectedCandidateId]);

  // Load cutouts whenever selected candidate changes
  useEffect(() => {
    if (!selectedCandidate) {
      setCutoutSequence(null);
      return;
    }

    if (cutoutSequence?.candidate_id === selectedCandidate.candidate_id) {
      return;
    }

    setActiveEpochIndex(0);

    if (isOffline) {
      setCutoutSequence(buildOfflineCutoutSequence(selectedCandidate));
      setIsLoadingCutouts(false);
      return;
    }

    const controller = new AbortController();
    setIsLoadingCutouts(true);

    void fetchCandidateCutouts(selectedCandidate.candidate_id, controller.signal)
      .then((seq) => {
        if (!controller.signal.aborted) {
          setCutoutSequence(seq);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setCutoutSequence(buildOfflineCutoutSequence(selectedCandidate));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoadingCutouts(false);
        }
      });

    return () => controller.abort();
  }, [selectedCandidate, isOffline, cutoutSequence?.candidate_id]);

  // Handle triage disposition with optimistic update & auto-advance
  const handleTriage = async (disposition: TriageDisposition, notes?: string | null): Promise<void> => {
    if (!selectedCandidate) return;
    const candId = selectedCandidate.candidate_id;

    onSetDisposition(candId, disposition);

    // Call REST endpoint if not in pure offline simulation mode
    if (!isOffline) {
      setIsSubmittingTriage(true);
      try {
        await submitCandidateTriage(candId, {
          disposition,
          notes: notes ?? null,
        });
      } catch {
        // Optimistic state kept for smooth triage, could trigger toast
      } finally {
        setIsSubmittingTriage(false);
      }
    }

    // Auto-advance to next candidate in queue
    if (autoAdvance) {
      const currentIndex = candidates.findIndex((c) => c.candidate_id === candId);
      if (currentIndex >= 0 && currentIndex + 1 < candidates.length) {
        const nextCandidate = candidates[currentIndex + 1];
        if (nextCandidate) {
          onSelectCandidate(nextCandidate.candidate_id);
        }
      }
    }
  };

  // Keyboard shortcut listener (1=Real, 2=Follow, 3=Bogus, 0=Reset, Up/Down navigation)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent): void => {
      // Ignore if focus is in an input or textarea
      if (
        document.activeElement instanceof HTMLInputElement ||
        document.activeElement instanceof HTMLTextAreaElement ||
        document.activeElement instanceof HTMLSelectElement
      ) {
        return;
      }

      if (e.key === "1" || e.key === "r") {
        e.preventDefault();
        void handleTriage("confirmed_real");
      } else if (e.key === "2" || e.key === "f") {
        e.preventDefault();
        void handleTriage("follow_up");
      } else if (e.key === "3" || e.key === "b") {
        e.preventDefault();
        void handleTriage("flagged_bogus");
      } else if (e.key === "0" || e.key === "u") {
        e.preventDefault();
        void handleTriage("unreviewed");
      } else if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        if (!selectedCandidate) return;
        const idx = candidates.findIndex((c) => c.candidate_id === selectedCandidate.candidate_id);
        if (idx >= 0 && idx + 1 < candidates.length) {
          const next = candidates[idx + 1];
          if (next) onSelectCandidate(next.candidate_id);
        }
      } else if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        if (!selectedCandidate) return;
        const idx = candidates.findIndex((c) => c.candidate_id === selectedCandidate.candidate_id);
        if (idx > 0) {
          const prev = candidates[idx - 1];
          if (prev) onSelectCandidate(prev.candidate_id);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedCandidate, candidates, autoAdvance, isOffline]);

  const currentDisposition = selectedCandidate
    ? (dispositions[selectedCandidate.candidate_id] ?? "unreviewed")
    : "unreviewed";

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-12 h-full">
      {/* Left Column (5 Cols): 3-Channel Blink-Loop Inspector & Triage Action Bar */}
      <div className="flex flex-col gap-3.5 xl:col-span-6 2xl:col-span-5">
        <ThreeChannelBlinkViewer
          cutoutSequence={cutoutSequence}
          isLoading={isLoadingCutouts}
          activeEpochIndex={activeEpochIndex}
          onEpochChange={setActiveEpochIndex}
        />

        <AcidTriageActionBar
          candidateId={selectedCandidate?.candidate_id ?? null}
          disposition={currentDisposition}
          isSubmitting={isSubmittingTriage}
          autoAdvance={autoAdvance}
          onSetDisposition={handleTriage}
          onToggleAutoAdvance={setAutoAdvance}
        />

        <CandidateDetailPanel
          candidate={selectedCandidate}
          cutoutSequence={cutoutSequence}
          activeEpochIndex={activeEpochIndex}
          onSelectEpoch={setActiveEpochIndex}
        />
      </div>

      {/* Right Column (7 Cols): Candidate Triage Queue Table */}
      <div className="xl:col-span-6 2xl:col-span-7">
        <CandidateTriageTable
          candidates={candidates}
          selectedCandidateId={selectedCandidate?.candidate_id ?? null}
          dispositions={dispositions}
          filters={filters}
          onSelectCandidate={onSelectCandidate}
          onFilterChange={onFilterChange}
        />
      </div>
    </div>
  );
};

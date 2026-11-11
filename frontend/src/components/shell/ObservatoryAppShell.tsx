import React, { useState } from "react";
import { CollapsibleNavRail, type ActiveNavView } from "./CollapsibleNavRail";
import { TopTelemetryHeader } from "./TopTelemetryHeader";
import type { ConnectionMode, DetectionSummaryResponse, PipelineStatusResponse } from "../../types/api";
import type { WsConnectionStatus } from "../../services/wsJobsClient";

export interface ObservatoryAppShellProps {
  readonly status: PipelineStatusResponse | null;
  readonly summary: DetectionSummaryResponse | null;
  readonly connectionMode: ConnectionMode;
  readonly wsStatus: WsConnectionStatus;
  readonly isRefreshing: boolean;
  readonly lastUpdatedIso: string | null;
  readonly activeView: ActiveNavView;
  readonly triageQueueCount?: number;
  readonly authEnabled?: boolean;
  readonly operatorUsername?: string | null;
  readonly onRefresh: () => void;
  readonly onReconnectWs?: () => void;
  readonly onResetDuckDb?: () => void;
  readonly onSelectView: (view: ActiveNavView) => void;
  readonly children: React.ReactNode;
}

export const ObservatoryAppShell: React.FC<ObservatoryAppShellProps> = ({
  status,
  summary,
  connectionMode,
  wsStatus,
  isRefreshing,
  lastUpdatedIso,
  activeView,
  triageQueueCount = 0,
  authEnabled = false,
  operatorUsername = null,
  onRefresh,
  onReconnectWs,
  onResetDuckDb,
  onSelectView,
  children,
}) => {
  const [isNavCollapsed, setIsNavCollapsed] = useState<boolean>(false);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#090a0f] text-slate-100 select-none">
      {/* Top Telemetry Header */}
      <TopTelemetryHeader
        status={status}
        summary={summary}
        connectionMode={connectionMode}
        wsStatus={wsStatus}
        isRefreshing={isRefreshing}
        lastUpdatedIso={lastUpdatedIso}
        authEnabled={authEnabled}
        operatorUsername={operatorUsername}
        onRefresh={onRefresh}
        onReconnectWs={onReconnectWs}
        onResetDuckDb={onResetDuckDb}
      />

      {/* Main Workspace Frame: Nav Rail + Content Pane */}
      <div className="flex flex-1 overflow-hidden">
        <CollapsibleNavRail
          isCollapsed={isNavCollapsed}
          activeView={activeView}
          triageQueueCount={triageQueueCount}
          onToggleCollapse={() => setIsNavCollapsed((prev) => !prev)}
          onSelectView={onSelectView}
        />

        <main className="flex-1 overflow-y-auto overflow-x-hidden p-3 sm:p-4 2xl:p-6 bg-[#090a0f]">
          <div className="mx-auto h-full max-w-[3600px]">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

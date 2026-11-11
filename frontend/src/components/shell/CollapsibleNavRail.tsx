import React from "react";
import { CheckCircle2, ChevronLeft, ChevronRight, LayoutDashboard, Settings, Database } from "lucide-react";

export type ActiveNavView = "triage" | "pipeline" | "settings";

export interface CollapsibleNavRailProps {
  readonly isCollapsed: boolean;
  readonly activeView: ActiveNavView;
  readonly triageQueueCount?: number;
  readonly onToggleCollapse: () => void;
  readonly onSelectView: (view: ActiveNavView) => void;
}

interface NavItem {
  readonly id: ActiveNavView;
  readonly label: string;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly badge?: number;
}

export const CollapsibleNavRail: React.FC<CollapsibleNavRailProps> = ({
  isCollapsed,
  activeView,
  triageQueueCount = 0,
  onToggleCollapse,
  onSelectView,
}) => {
  const items: readonly NavItem[] = [
    {
      id: "triage",
      label: "Candidate Triage",
      icon: CheckCircle2,
      badge: triageQueueCount > 0 ? triageQueueCount : undefined,
    },
    {
      id: "pipeline",
      label: "Live Pipeline",
      icon: LayoutDashboard,
    },
    {
      id: "settings",
      label: "Global Settings",
      icon: Settings,
    },
  ];

  return (
    <aside
      aria-label="Observatory Navigation Rail"
      className={`flex flex-col border-r border-[#26293d] bg-[#11131c] transition-all duration-200 z-20 shrink-0 ${
        isCollapsed ? "w-14" : "w-60 lg:w-64"
      }`}
    >
      {/* Top Toggle Header */}
      <div className="flex min-h-12 h-auto items-center justify-between border-b border-[#26293d] px-3.5 py-2">
        {!isCollapsed && (
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 truncate">
            Workspaces
          </span>
        )}
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label={isCollapsed ? "Expand navigation rail" : "Collapse navigation rail"}
          className="rounded p-1 text-slate-400 hover:bg-[#181b28] hover:text-slate-200 shrink-0"
        >
          {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      {/* Nav List */}
      <nav className="flex-1 space-y-1 p-2 overflow-y-auto">
        {items.map((item) => {
          const isActive = activeView === item.id;
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectView(item.id)}
              title={isCollapsed ? item.label : undefined}
              className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-xs font-medium transition-colors ${
                isActive
                  ? "border border-[#581c87] bg-[#24103a] text-purple-100 font-semibold shadow-inner"
                  : "border border-transparent text-slate-300 hover:border-[#3b185f] hover:bg-[#181b28] hover:text-slate-100"
              } ${isCollapsed ? "justify-center px-0" : ""}`}
            >
              <Icon className={`h-4 w-4 shrink-0 ${isActive ? "text-purple-300" : "text-slate-400"}`} />
              {!isCollapsed && (
                <span className="flex-1 text-left whitespace-normal leading-tight">{item.label}</span>
              )}
              {!isCollapsed && item.badge !== undefined && (
                <span className="rounded-full bg-[#581c87] px-2 py-0.5 text-[10px] font-mono text-purple-100 font-bold tabular-nums shrink-0">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      {!isCollapsed && (
        <div className="border-t border-[#26293d] p-3 text-[11px] text-slate-400 space-y-1">
          <div className="flex items-center gap-1.5 font-mono text-[10px] text-purple-300">
            <Database className="h-3 w-3 shrink-0" />
            <span className="truncate">OLTP + OLAP Polyglot</span>
          </div>
          <div className="truncate">Observatory Console</div>
        </div>
      )}
    </aside>
  );
};

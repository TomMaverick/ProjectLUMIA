import React, { useEffect, useRef, useState } from "react";
import { Terminal, ArrowDown, Search, ShieldAlert } from "lucide-react";
import type { WsTaskLogAppendedPayload } from "../../types/api";

export interface TaskLogStreamConsoleProps {
  readonly logs: readonly WsTaskLogAppendedPayload[];
  readonly droppedLogFrames?: number;
}

const SEVERITY_COLORS: Record<string, string> = {
  DEBUG: "text-slate-500",
  INFO: "text-purple-300",
  WARNING: "text-amber-300",
  ERROR: "text-rose-400 font-bold",
  CRITICAL: "text-rose-500 font-extrabold bg-rose-950/40",
};

export const TaskLogStreamConsole: React.FC<TaskLogStreamConsoleProps> = ({
  logs,
  droppedLogFrames = 0,
}) => {
  const [minSeverity, setMinSeverity] = useState<"ALL" | "INFO" | "WARNING" | "ERROR">("ALL");
  const [filterQuery, setFilterQuery] = useState<string>("");
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const filteredLogs = logs.filter((log) => {
    if (minSeverity === "INFO" && log.severity === "DEBUG") return false;
    if (minSeverity === "WARNING" && (log.severity === "DEBUG" || log.severity === "INFO")) return false;
    if (minSeverity === "ERROR" && log.severity !== "ERROR" && log.severity !== "CRITICAL") return false;

    if (filterQuery.trim()) {
      const q = filterQuery.toLowerCase();
      const matchMsg = log.message.toLowerCase().includes(q);
      const matchCode = log.event_code ? log.event_code.toLowerCase().includes(q) : false;
      return matchMsg || matchCode;
    }
    return true;
  });

  useEffect(() => {
    if (autoScroll && bottomRef.current && typeof bottomRef.current.scrollIntoView === "function") {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs.length, autoScroll]);

  return (
    <div className="flex flex-col h-[360px] rounded-lg border border-[#3b185f] bg-[#090a0f] shadow-inner font-mono text-xs">
      {/* Console Chrome */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#26293d] bg-[#11131c] px-3 py-2">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-purple-400" />
          <span className="font-bold text-slate-200 uppercase tracking-wider text-[11px]">
            Live Task Log Stream ({filteredLogs.length} entries)
          </span>
        </div>

        {droppedLogFrames > 0 && (
          <div className="flex items-center gap-1 rounded bg-amber-950/60 border border-amber-700/60 px-2 py-0.5 text-[10px] text-amber-300">
            <ShieldAlert className="h-3 w-3" />
            <span>Dropped Frames: {droppedLogFrames}</span>
          </div>
        )}

        <div className="flex items-center gap-2">
          {/* Severity filter */}
          <select
            value={minSeverity}
            onChange={(e) => setMinSeverity(e.target.value as "ALL" | "INFO" | "WARNING" | "ERROR")}
            className="rounded border border-[#3b185f] bg-[#090a0f] px-2 py-0.5 text-[10px] text-slate-300 focus:outline-none"
          >
            <option value="ALL">Severity: ALL</option>
            <option value="INFO">Severity: &gt;= INFO</option>
            <option value="WARNING">Severity: &gt;= WARNING</option>
            <option value="ERROR">Severity: ERRORS ONLY</option>
          </select>

          {/* Search filter */}
          <div className="relative">
            <Search className="absolute left-2 top-1.5 h-3 w-3 text-slate-500" />
            <input
              type="text"
              placeholder="Filter events..."
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              className="rounded border border-[#3b185f] bg-[#090a0f] pl-6 pr-2 py-0.5 text-[10px] text-slate-200 placeholder-slate-600 focus:outline-none"
            />
          </div>

          <button
            type="button"
            onClick={() => setAutoScroll((prev) => !prev)}
            className={`rounded border px-2 py-0.5 text-[10px] font-semibold flex items-center gap-1 ${
              autoScroll
                ? "border-[#581c87] bg-[#581c87] text-white"
                : "border-[#26293d] bg-[#181b28] text-slate-400"
            }`}
          >
            <ArrowDown className="h-3 w-3" />
            <span>AutoScroll</span>
          </button>
        </div>
      </div>

      {/* Log list terminal */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1 select-text">
        {filteredLogs.length === 0 ? (
          <div className="text-slate-600 italic text-[11px] py-4 text-center">
            No log entries matching filter criteria.
          </div>
        ) : (
          filteredLogs.map((log) => {
            const time = log.logged_at_utc.slice(11, 19);
            const sevColor = SEVERITY_COLORS[log.severity] ?? "text-slate-300";

            return (
              <div key={log.log_id} className="leading-tight text-[11px] hover:bg-[#11131c]/60 px-1 rounded">
                <span className="text-slate-500 mr-2">{time}</span>
                <span className={`font-bold mr-2 ${sevColor}`}>[{log.severity}]</span>
                {log.event_code && (
                  <span className="text-purple-400 font-semibold mr-2">[{log.event_code}]</span>
                )}
                <span className="text-slate-300">{log.message}</span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

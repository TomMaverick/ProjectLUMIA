import React, { useState } from "react";
import { Shield, Database, Sliders, Save, Check, Server, HardDrive } from "lucide-react";
import { DuckDbConfigCard } from "./DuckDbConfigCard";
import { PostgresConfigCard } from "./PostgresConfigCard";
import type { GlobalProgramSettings } from "../../types/settings";

export interface GlobalOptionsViewProps {
  readonly settings: GlobalProgramSettings;
  readonly onSaveSettings: (settings: GlobalProgramSettings) => void;
}

export const GlobalOptionsView: React.FC<GlobalOptionsViewProps> = ({
  settings: initialSettings,
  onSaveSettings,
}) => {
  const [activeTab, setActiveTab] = useState<"databases" | "storage" | "connection" | "security" | "display">("databases");
  const [currentSettings, setCurrentSettings] = useState<GlobalProgramSettings>(initialSettings);
  const [showSavedToast, setShowSavedToast] = useState<boolean>(false);

  const handleSave = (): void => {
    onSaveSettings(currentSettings);
    try {
      localStorage.setItem("lumia.ui.settings.v1", JSON.stringify(currentSettings));
      if (currentSettings.backend_connection === "remote") {
        localStorage.setItem("lumia.ui.remote_base_url", currentSettings.remote_base_url);
      }
    } catch {
      // storage unavailable
    }
    setShowSavedToast(true);
    setTimeout(() => setShowSavedToast(false), 2500);
  };

  return (
    <div className="flex flex-col gap-4 max-w-[1600px] mx-auto pb-8">
      {/* View Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#26293d] pb-3">
        <div>
          <h2 className="text-base font-bold text-slate-100 uppercase tracking-wider">
            Global Program Configuration &amp; Settings
          </h2>
          <p className="text-xs text-slate-400">
            Manage Polyglot Persistence, Deployment Target, Operator Authentication and Display Ergonomics
          </p>
        </div>

        <div className="flex items-center gap-2">
          {showSavedToast && (
            <span className="inline-flex items-center gap-1 text-xs text-emerald-300 font-mono animate-fade-in">
              <Check className="h-3.5 w-3.5" />
              <span>Settings Applied</span>
            </span>
          )}
          <button
            type="button"
            onClick={handleSave}
            className="inline-flex items-center gap-1.5 rounded border border-[#581c87] bg-[#3b185f] px-4 py-1.5 text-xs font-bold text-purple-100 hover:bg-[#581c87] transition-colors shadow-md"
          >
            <Save className="h-3.5 w-3.5" />
            <span>Save &amp; Apply</span>
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-[#26293d] pb-1">
        <button
          type="button"
          onClick={() => setActiveTab("databases")}
          className={`flex items-center gap-2 rounded-t px-3.5 py-2 text-xs font-semibold transition-colors ${
            activeTab === "databases"
              ? "border-b-2 border-[#a855f7] bg-[#181b28] text-purple-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Database className="h-3.5 w-3.5 text-purple-400" />
          <span>Polyglot Databases (OLAP &amp; OLTP)</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("storage")}
          className={`flex items-center gap-2 rounded-t px-3.5 py-2 text-xs font-semibold transition-colors ${
            activeTab === "storage"
              ? "border-b-2 border-[#a855f7] bg-[#181b28] text-purple-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <HardDrive className="h-3.5 w-3.5 text-purple-400" />
          <span>Data Archive &amp; Storage</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("connection")}
          className={`flex items-center gap-2 rounded-t px-3.5 py-2 text-xs font-semibold transition-colors ${
            activeTab === "connection"
              ? "border-b-2 border-[#a855f7] bg-[#181b28] text-purple-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Server className="h-3.5 w-3.5 text-purple-400" />
          <span>Backend Connection</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("security")}
          className={`flex items-center gap-2 rounded-t px-3.5 py-2 text-xs font-semibold transition-colors ${
            activeTab === "security"
              ? "border-b-2 border-[#a855f7] bg-[#181b28] text-purple-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Shield className="h-3.5 w-3.5 text-purple-400" />
          <span>Authentication &amp; Security</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("display")}
          className={`flex items-center gap-2 rounded-t px-3.5 py-2 text-xs font-semibold transition-colors ${
            activeTab === "display"
              ? "border-b-2 border-[#a855f7] bg-[#181b28] text-purple-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Sliders className="h-3.5 w-3.5 text-purple-400" />
          <span>Observatory Ergonomics</span>
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === "databases" && (
        <div className="space-y-4">
          <DuckDbConfigCard
            config={currentSettings.duckdb}
            onChange={(nextDuckDb) =>
              setCurrentSettings({ ...currentSettings, duckdb: nextDuckDb })
            }
          />

          <PostgresConfigCard
            config={currentSettings.postgres}
            onChange={(nextPostgres) =>
              setCurrentSettings({ ...currentSettings, postgres: nextPostgres })
            }
          />
        </div>
      )}

      {activeTab === "storage" && (
        <div className="rounded-lg border border-[#3b185f] bg-[#11131c] p-5 shadow-lg shadow-black/40 space-y-4 text-xs">
          <div className="border-b border-[#26293d] pb-3">
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Telescope FITS Archive Storage
            </h3>
            <p className="text-[11px] text-slate-400">
              Configure the filesystem directory path where raw and calibrated telescope FITS frames are archived
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Telescope FITS Archive Path
              </label>
              <input
                type="text"
                value={currentSettings.archive_storage_path ?? "data/archives/"}
                onChange={(e) =>
                  setCurrentSettings({
                    ...currentSettings,
                    archive_storage_path: e.target.value,
                  })
                }
                placeholder="data/archives/"
                className="w-full rounded border border-[#3b185f] bg-[#090a0f] px-3 py-2 text-slate-200 font-mono text-xs focus:outline-none focus:border-[#a855f7]"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">
                Supports relative paths (e.g., data/archives/) and absolute paths (e.g., /mnt/telescope_nas/fits or E:\Observatory\FITS).
                The directory is automatically created by the backend if it does not exist.
              </span>
            </div>

            <div className="rounded border border-[#26293d] bg-[#090a0f] p-3 text-[11px] text-slate-400 space-y-1">
              <div className="font-semibold text-slate-300">Storage &amp; Ingestion Notes:</div>
              <div>• Science and reference images are loaded from this directory by the Astrometric FITS Ingestion Gatekeeper.</div>
              <div>• Ensure the Project LUMIA backend process has read and write permissions on this filesystem path.</div>
              <div>• Changes take effect immediately upon clicking "Save &amp; Apply".</div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "connection" && (
        <div className="rounded-lg border border-[#3b185f] bg-[#11131c] p-5 shadow-lg shadow-black/40 space-y-4">
          <div className="border-b border-[#26293d] pb-3">
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Backend Connection &amp; Target Host
            </h3>
            <p className="text-xs text-slate-400">
              Configure target Project LUMIA API host for local workstation operation or remote observatory node
            </p>
          </div>

          <div className="space-y-3">
            <label className="flex items-start gap-3 rounded-md border border-[#26293d] bg-[#090a0f] p-4 cursor-pointer">
              <input
                type="radio"
                name="backend_connection"
                value="local"
                checked={currentSettings.backend_connection === "local"}
                onChange={() =>
                  setCurrentSettings({
                    ...currentSettings,
                    backend_connection: "local",
                  })
                }
                className="mt-1 accent-[#581c87]"
              />
              <div>
                <span className="font-bold text-slate-200 block text-xs">
                  Local Device (Standard)
                </span>
                <span className="text-[11px] text-slate-400 mt-1 block leading-relaxed">
                  Direct connection via local Vite proxy (/api routed to http://127.0.0.1:8000). Ideal for single-node development and standalone observatory workstations.
                </span>
              </div>
            </label>

            <label className="flex items-start gap-3 rounded-md border border-[#26293d] bg-[#090a0f] p-4 cursor-pointer">
              <input
                type="radio"
                name="backend_connection"
                value="remote"
                checked={currentSettings.backend_connection === "remote"}
                onChange={() =>
                  setCurrentSettings({
                    ...currentSettings,
                    backend_connection: "remote",
                  })
                }
                className="mt-1 accent-[#581c87]"
              />
              <div className="flex-1">
                <span className="font-bold text-slate-200 block text-xs">
                  Remote Server
                </span>
                <span className="text-[11px] text-slate-400 mt-1 block leading-relaxed">
                  Connect to an external Project LUMIA server running on an observatory LAN, telescope mount controller, or cloud cluster.
                </span>

                {currentSettings.backend_connection === "remote" && (
                  <div className="mt-3 pt-3 border-t border-[#26293d]">
                    <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                      Remote Server Base URL
                    </label>
                    <input
                      type="url"
                      value={currentSettings.remote_base_url}
                      onChange={(e) =>
                        setCurrentSettings({
                          ...currentSettings,
                          remote_base_url: e.target.value,
                        })
                      }
                      placeholder="http://192.168.1.100:8000"
                      className="w-full rounded border border-[#3b185f] bg-[#11131c] px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-[#a855f7]"
                    />
                    <span className="text-[10px] text-slate-500 mt-1 block">
                      Ensure CORS is enabled on the remote server for this frontend origin.
                    </span>
                  </div>
                )}
              </div>
            </label>
          </div>
        </div>
      )}

      {activeTab === "security" && (
        <div className="rounded-lg border border-[#3b185f] bg-[#11131c] p-5 shadow-lg shadow-black/40 space-y-4">
          <div className="border-b border-[#26293d] pb-3">
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Feature-Toggled Operator Authentication
            </h3>
            <p className="text-xs text-slate-400">
              Control session protection mode according to deployment environment (Single-User Dome vs. Multi-User Facility)
            </p>
          </div>

          <div className="rounded-md border border-[#26293d] bg-[#090a0f] p-4">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={currentSettings.auth_enabled}
                onChange={(e) =>
                  setCurrentSettings({ ...currentSettings, auth_enabled: e.target.checked })
                }
                className="mt-1 accent-[#581c87]"
              />
              <div>
                <span className="font-bold text-slate-200 block text-xs">
                  Require Operator Authentication (JWT Bearer + HttpOnly Cookie)
                </span>
                <span className="text-[11px] text-slate-400 mt-1 block leading-relaxed">
                  When enabled, all REST endpoints and WebSockets enforce 15-minute JWT tokens with single-use tickets and role checks.
                  When disabled (Local Observatory Bypass), authentication screens are bypassed automatically for local single-user dome operations.
                </span>
              </div>
            </label>
          </div>

          <div className="text-[11px] text-slate-400 border-l-2 border-[#581c87] pl-3 py-1 font-mono">
            Security mode: {currentSettings.auth_enabled ? "ENTERPRISE RBAC (ENFORCED)" : "LOCAL BYPASS (AUTH SCREEN DISABLED)"}
          </div>
        </div>
      )}

      {activeTab === "display" && (
        <div className="rounded-lg border border-[#3b185f] bg-[#11131c] p-5 shadow-lg shadow-black/40 space-y-4 text-xs">
          <div className="border-b border-[#26293d] pb-3">
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Display Density &amp; Blink Defaults
            </h3>
            <p className="text-[11px] text-slate-400">
              Customize ergonomics to fit night-vision requirements across 720p, 1080p, 1440p and 4K viewports
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Root Viewport UI Scale
              </label>
              <select
                value={currentSettings.display.ui_scale_percent}
                onChange={(e) =>
                  setCurrentSettings({
                    ...currentSettings,
                    display: {
                      ...currentSettings.display,
                      ui_scale_percent: Number(e.target.value) as 100 | 110 | 120 | 130 | 150,
                    },
                  })
                }
                className="w-full rounded border border-[#3b185f] bg-[#090a0f] p-2 text-slate-200 focus:outline-none"
              >
                <option value={100}>100% (Standard Viewport)</option>
                <option value={110}>110% (Comfortable 1080p / 1440p)</option>
                <option value={120}>120% (Compact 4K Display)</option>
                <option value={130}>130% (High Density 4K UHD)</option>
                <option value={150}>150% (Large / Remote Wall Display)</option>
              </select>
            </div>

            <div>
              <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Default Blink Cadence
              </label>
              <select
                value={currentSettings.display.default_cadence_hz}
                onChange={(e) =>
                  setCurrentSettings({
                    ...currentSettings,
                    display: {
                      ...currentSettings.display,
                      default_cadence_hz: Number(e.target.value),
                    },
                  })
                }
                className="w-full rounded border border-[#3b185f] bg-[#090a0f] p-2 text-slate-200 focus:outline-none"
              >
                <option value={1}>1 Hz (1000 ms per epoch)</option>
                <option value={2}>2 Hz (500 ms - Recommended)</option>
                <option value={4}>4 Hz (250 ms)</option>
                <option value={8}>8 Hz (125 ms)</option>
              </select>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

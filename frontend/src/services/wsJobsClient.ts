/**
 * WebSocket Client for streaming pipeline jobs, task state, metrics & logs
 * from /api/v1/ws/jobs per Module 10 specification.
 */

import type { WsEnvelope, WebSocketTicketResponse } from "../types/api";
import { getAccessToken, getApiBaseUrl, requestJson } from "./httpClient";

export type WsConnectionStatus = "disconnected" | "connecting" | "connected" | "reconnecting";

export type WsEventListener = (envelope: WsEnvelope) => void;
export type WsStatusListener = (status: WsConnectionStatus) => void;

export interface WsClientOptions {
  readonly jobId?: string | null;
  readonly autoReconnect?: boolean;
  readonly authEnabled?: boolean;
}

export class WsJobsClient {
  private ws: WebSocket | null = null;
  private status: WsConnectionStatus = "disconnected";
  private eventListeners: Set<WsEventListener> = new Set();
  private statusListeners: Set<WsStatusListener> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectAttempt = 0;
  private isDisposed = false;
  private subscribedJobId: string | null = null;
  private authEnabled = false;

  public constructor(options: WsClientOptions = {}) {
    this.subscribedJobId = options.jobId ?? null;
    this.authEnabled = options.authEnabled ?? false;
  }

  public getStatus(): WsConnectionStatus {
    return this.status;
  }

  public addEventListener(listener: WsEventListener): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  public addStatusListener(listener: WsStatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.status);
    return () => this.statusListeners.delete(listener);
  }

  public subscribeJob(jobId: string | null, minLogSeverity: "DEBUG" | "INFO" | "WARNING" | "ERROR" = "INFO"): void {
    this.subscribedJobId = jobId;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          action: "subscribe",
          job_id: jobId,
          min_log_severity: minLogSeverity,
        })
      );
    }
  }

  public async resetAndReconnect(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.reconnectAttempt = 0;
    this.isDisposed = false;
    await this.connect();
  }

  public async connect(): Promise<void> {
    if (this.isDisposed || this.status === "connecting" || this.status === "connected") {
      return;
    }

    this.setStatus(this.reconnectAttempt > 0 ? "reconnecting" : "connecting");

    let ticket: string | null = null;
    if (this.authEnabled && getAccessToken()) {
      try {
        const ticketRes = await requestJson<WebSocketTicketResponse>("/api/v1/auth/ws-ticket", {
          method: "POST",
        });
        ticket = ticketRes.ticket;
      } catch {
        // If ticket minting fails, fallback or schedule reconnect
        this.scheduleReconnect();
        return;
      }
    }

    const baseUrl = getApiBaseUrl();
    let protocol: string;
    let host: string;

    if (baseUrl) {
      try {
        const parsed = new URL(baseUrl);
        protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
        host = parsed.host;
      } catch {
        const loc = window.location;
        protocol = loc.protocol === "https:" ? "wss:" : "ws:";
        host = loc.host;
      }
    } else {
      const loc = window.location;
      protocol = loc.protocol === "https:" ? "wss:" : "ws:";
      host = loc.host;
    }

    const params = new URLSearchParams();

    if (ticket) {
      params.set("ticket", ticket);
    }
    if (this.subscribedJobId) {
      params.set("job_id", this.subscribedJobId);
    }

    const query = params.toString() ? `?${params.toString()}` : "";
    const wsUrl = `${protocol}//${host}/api/v1/ws/jobs${query}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.reconnectAttempt = 0;
        this.setStatus("connected");
        this.startHeartbeat();
      };

      this.ws.onmessage = (event: MessageEvent<string>) => {
        try {
          const envelope = JSON.parse(event.data) as WsEnvelope;
          for (const listener of this.eventListeners) {
            listener(envelope);
          }
        } catch {
          // Ignore malformed message frame
        }
      };

      this.ws.onerror = () => {
        // Handled in onclose
      };

      this.ws.onclose = (event: CloseEvent) => {
        this.stopHeartbeat();
        this.ws = null;
        this.setStatus("disconnected");

        // Code 4401: Unauthorized ticket or closed intentionally
        if (event.code === 4401 || this.isDisposed) {
          return;
        }

        this.scheduleReconnect();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  public disconnect(): void {
    this.isDisposed = true;
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus("disconnected");
  }

  private setStatus(nextStatus: WsConnectionStatus): void {
    if (this.status !== nextStatus) {
      this.status = nextStatus;
      for (const listener of this.statusListeners) {
        listener(nextStatus);
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.isDisposed || this.reconnectTimer) {
      return;
    }

    this.reconnectAttempt += 1;
    const delayMs = Math.min(1000 * Math.pow(1.5, this.reconnectAttempt - 1), 15000);

    this.setStatus("reconnecting");
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect();
    }, delayMs);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(
          JSON.stringify({
            action: "ping",
            client_sent_at_utc: new Date().toISOString(),
          })
        );
      }
    }, 25000);
  }

  private stopHeartbeat(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}

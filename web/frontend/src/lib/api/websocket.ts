import type { WsMessage, WsProgress, WsRunComplete, WsStrategyDone } from '../types/index';

export interface WsCallbacks {
  onProgress?: (msg: WsProgress) => void;
  onStrategyDone?: (msg: WsStrategyDone) => void;
  onRunComplete?: (msg: WsRunComplete) => void;
  onError?: (err: Event | Error) => void;
  onClose?: () => void;
}

/**
 * WebSocket client for streaming simulation run progress.
 *
 * Connects to /api/ws/runs/{runId}, parses JSON messages, and routes
 * them to the appropriate callback. Auto-reconnects on unexpected
 * disconnect with exponential backoff.
 */
export class RunWebSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly maxReconnectAttempts = 5;
  private readonly baseDelay = 500; // ms

  constructor(
    private readonly runId: string,
    private readonly callbacks: WsCallbacks,
  ) {
    this.connect();
  }

  private buildUrl(): string {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/api/ws/runs/${encodeURIComponent(this.runId)}`;
  }

  private connect(): void {
    if (this.closed) return;

    const url = this.buildUrl();
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event: MessageEvent) => {
      this.reconnectAttempts = 0; // successful communication resets backoff
      try {
        const msg: WsMessage = JSON.parse(event.data as string);
        this.dispatch(msg);
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onerror = (event: Event) => {
      this.callbacks.onError?.(event);
    };

    this.ws.onclose = () => {
      if (this.closed) {
        this.callbacks.onClose?.();
        return;
      }
      this.scheduleReconnect();
    };
  }

  private dispatch(msg: WsMessage): void {
    if ('status' in msg && msg.status === 'run_complete') {
      this.callbacks.onRunComplete?.(msg as WsRunComplete);
      // Run is done, no need to reconnect after server closes the socket
      this.closed = true;
      return;
    }
    if ('percent' in msg) {
      this.callbacks.onProgress?.(msg as WsProgress);
      return;
    }
    if ('strategy' in msg && 'status' in msg) {
      this.callbacks.onStrategyDone?.(msg as WsStrategyDone);
      return;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.callbacks.onError?.(new Error('WebSocket: max reconnect attempts reached'));
      this.callbacks.onClose?.();
      return;
    }

    const delay = this.baseDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  close(): void {
    this.closed = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

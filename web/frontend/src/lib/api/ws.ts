import type { WsMessage, WsProgress, WsStrategyDone } from '../types';

export interface RunSocketHandlers {
  onProgress(msg: WsProgress): void;
  onStrategyDone(msg: WsStrategyDone): void;
  onRunComplete(): void;
  onError?(message: string): void;
}

/**
 * Live progress for one run. Reconnects a few times with backoff; the caller also polls the
 * summary, so a dead socket only costs latency, never correctness.
 */
export class RunSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private attempts = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly runId: string,
    private readonly h: RunSocketHandlers,
  ) {
    this.connect();
  }

  private connect(): void {
    if (this.closed) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/api/ws/runs/${encodeURIComponent(this.runId)}`);
    this.ws = ws;
    ws.onmessage = (ev: MessageEvent) => {
      this.attempts = 0;
      let msg: WsMessage;
      try {
        msg = JSON.parse(String(ev.data)) as WsMessage;
      } catch {
        return;
      }
      this.dispatch(msg);
    };
    ws.onclose = () => {
      if (this.closed || this.attempts >= 4) return;
      const delay = 500 * 2 ** this.attempts++;
      this.timer = setTimeout(() => this.connect(), delay);
    };
  }

  private dispatch(msg: WsMessage): void {
    if ('percent' in msg) {
      this.h.onProgress(msg);
    } else if ('strategy' in msg) {
      this.h.onStrategyDone(msg);
    } else if ('error' in msg) {
      this.h.onError?.(msg.error);
      this.close();
    } else if (msg.status === 'run_complete') {
      this.h.onRunComplete();
      this.close();
    }
  }

  close(): void {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.ws?.close();
    this.ws = null;
  }
}

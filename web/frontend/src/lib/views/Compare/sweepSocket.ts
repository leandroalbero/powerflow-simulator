import type { RunStatusValue, SweepCase, WsSweepMessage } from '../../types';

/** The shared message type lacks the error the backend sends for an unknown sweep id. */
type Message = WsSweepMessage | { error: string };

export interface SweepSocketHandlers {
  onCase(c: SweepCase): void;
  onProgress(done: number, total: number): void;
  onComplete(status: RunStatusValue): void;
  onError(message: string): void;
}

/**
 * Live progress for one sweep. The backend replays finished cases on connect, so a reconnect
 * loses nothing; the caller also polls while the socket is quiet.
 */
export class SweepSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private attempts = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  lastMessageAt = 0;

  constructor(
    private readonly sweepId: string,
    private readonly h: SweepSocketHandlers,
  ) {
    this.connect();
  }

  private connect(): void {
    if (this.closed) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws: WebSocket;
    try {
      ws = new WebSocket(`${proto}//${location.host}/api/ws/sweep/${encodeURIComponent(this.sweepId)}`);
    } catch {
      return;
    }
    this.ws = ws;
    ws.onmessage = (ev: MessageEvent) => {
      this.attempts = 0;
      this.lastMessageAt = Date.now();
      let msg: Message;
      try {
        msg = JSON.parse(String(ev.data)) as Message;
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

  private dispatch(msg: Message): void {
    if ('error' in msg) {
      this.h.onError(msg.error);
      this.close();
    } else if (msg.type === 'case_complete') {
      this.h.onCase(msg.case);
    } else if (msg.type === 'progress') {
      this.h.onProgress(msg.done, msg.total);
    } else if (msg.type === 'sweep_complete') {
      this.h.onComplete(msg.status);
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

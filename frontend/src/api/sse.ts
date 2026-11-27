// Minimal Server-Sent Events parser for streams that arrive over `fetch`
// (EventSource only supports GET). Follows the WHATWG event-stream rules:
// lines end in CRLF, LF or CR; `field: value` with one optional space;
// multiple `data:` lines are joined with "\n"; a blank line dispatches;
// lines starting with ":" are comments. Chunks may split anywhere,
// including inside a multi-byte character or between "\r" and "\n".

export interface SSEMessage {
  event: string;
  data: string;
  id: string | null;
}

const EOL = /\r\n|\r|\n/;

export class SSEParser {
  private buffer = "";
  private eventType = "";
  private dataLines: string[] = [];
  private lastId: string | null = null;
  private sawBom = false;

  /** Feed decoded text; returns every event completed by this chunk. */
  feed(chunk: string): SSEMessage[] {
    if (!this.sawBom) {
      if (chunk.length === 0) return [];
      if (chunk.charCodeAt(0) === 0xfeff) chunk = chunk.slice(1);
      this.sawBom = true;
    }
    this.buffer += chunk;
    const out: SSEMessage[] = [];
    for (;;) {
      const match = EOL.exec(this.buffer);
      if (!match) break;
      // A trailing lone "\r" may be the first half of "\r\n"; wait for more.
      if (match[0] === "\r" && match.index === this.buffer.length - 1) break;
      const line = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      this.processLine(line, out);
    }
    return out;
  }

  /** Call at end of stream. Dispatches a final event left without a blank line. */
  flush(): SSEMessage[] {
    const out: SSEMessage[] = [];
    if (this.buffer.length > 0) {
      const rest = this.buffer.replace(/\r$/, "");
      this.buffer = "";
      this.processLine(rest, out);
    }
    this.dispatch(out);
    return out;
  }

  private processLine(line: string, out: SSEMessage[]) {
    if (line === "") {
      this.dispatch(out);
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    let field: string;
    let value: string;
    if (colon === -1) {
      field = line;
      value = "";
    } else {
      field = line.slice(0, colon);
      value = line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
    }
    switch (field) {
      case "event":
        this.eventType = value;
        break;
      case "data":
        this.dataLines.push(value);
        break;
      case "id":
        if (!value.includes("\0")) this.lastId = value;
        break;
      default:
        // "retry" and unknown fields are irrelevant for a one-shot stream.
        break;
    }
  }

  private dispatch(out: SSEMessage[]) {
    if (this.dataLines.length === 0) {
      this.eventType = "";
      return;
    }
    out.push({
      event: this.eventType || "message",
      data: this.dataLines.join("\n"),
      id: this.lastId,
    });
    this.eventType = "";
    this.dataLines = [];
  }
}

/**
 * Read an SSE response body to completion, invoking `onMessage` for each event.
 * Resolves when the stream ends; rejects on network errors or abort.
 */
export async function readSSE(
  body: ReadableStream<Uint8Array>,
  onMessage: (msg: SSEMessage) => void,
  signal?: AbortSignal,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = new SSEParser();
  const onAbort = () => {
    void reader.cancel().catch(() => undefined);
  };
  signal?.addEventListener("abort", onAbort);
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const msg of parser.feed(decoder.decode(value, { stream: true }))) onMessage(msg);
    }
    for (const msg of parser.feed(decoder.decode())) onMessage(msg);
    for (const msg of parser.flush()) onMessage(msg);
  } finally {
    signal?.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
}

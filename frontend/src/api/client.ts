import { readSSE } from "./sse";
import type {
  AnalysisReport,
  ChatEvent,
  Conversation,
  CreateVideoRequest,
  CreateVideoResponse,
  IngestEvent,
  IngestStep,
  Message,
  ObjectSelection,
  Timeline,
  ToolRun,
  Video,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Pull a human-readable message out of an error response body. */
async function errorFromResponse(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`.trim();
  try {
    const text = await res.text();
    if (text) {
      try {
        const body: unknown = JSON.parse(text);
        if (body && typeof body === "object") {
          const b = body as Record<string, unknown>;
          const candidate = b.detail ?? b.message ?? b.error;
          if (typeof candidate === "string") message = candidate;
          else if (candidate) message = JSON.stringify(candidate);
        }
      } catch {
        message = text.slice(0, 300);
      }
    }
  } catch {
    // keep status text
  }
  return new ApiError(res.status, message);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "Can't reach the server. Check that the backend is running on port 8010.");
  }
  if (!res.ok) throw await errorFromResponse(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const json = (body: unknown) => JSON.stringify(body);
const enc = encodeURIComponent;

export const api = {
  // Videos
  listVideos: () => request<Video[]>("/videos"),
  getVideo: (id: string) => request<Video>(`/videos/${enc(id)}`),
  createVideo: (body: CreateVideoRequest) =>
    request<CreateVideoResponse>("/videos", { method: "POST", body: json(body) }),
  ingestVideo: (id: string) => request<Video>(`/videos/${enc(id)}/ingest`, { method: "POST" }),
  deleteVideo: (id: string) => request<void>(`/videos/${enc(id)}`, { method: "DELETE" }),
  getSteps: (id: string) => request<IngestStep[]>(`/videos/${enc(id)}/steps`),
  getTimeline: (id: string) => request<Timeline>(`/videos/${enc(id)}/timeline`),

  // Analysis + tools
  runAnalysis: (id: string, refresh = false) =>
    request<AnalysisReport>(`/videos/${enc(id)}/analysis`, {
      method: "POST",
      body: json(refresh ? { refresh: true } : {}),
    }),
  listToolRuns: (id: string) => request<ToolRun[]>(`/videos/${enc(id)}/tool-runs`),
  /** Click-to-ask: outline the object at (x, y) (0..1 of the frame) on the frame at t. */
  selectObject: (id: string, t: number, x: number, y: number) =>
    request<ObjectSelection>(`/videos/${enc(id)}/select`, { method: "POST", body: json({ t, x, y }) }),
  getToolRun: (id: string) => request<ToolRun>(`/tool-runs/${enc(id)}`),

  // Conversations
  createConversation: (videoId: string) =>
    request<Conversation>(`/videos/${enc(videoId)}/conversations`, { method: "POST" }),
  listConversations: (videoId: string) =>
    request<Conversation[]>(`/videos/${enc(videoId)}/conversations`),
  listMessages: (conversationId: string) =>
    request<Message[]>(`/conversations/${enc(conversationId)}/messages`),
};

// ---- Streaming -------------------------------------------------------------

const CHAT_EVENTS = new Set<ChatEvent["type"]>([
  "user",
  "plan",
  "tool_call",
  "tool_result",
  "answer_delta",
  "final",
  "error",
]);

/**
 * POST a message and stream the agent's reply. SSE arrives over the POST
 * response, so this uses fetch + a stream reader rather than EventSource.
 */
export async function streamMessage(
  conversationId: string,
  content: string,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
  selectionId?: string | null,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/conversations/${enc(conversationId)}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: json(selectionId ? { content, selection_id: selectionId } : { content }),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "Can't reach the server. Check that the backend is running on port 8010.");
  }
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError(res.status, "The server returned an empty stream.");

  await readSSE(
    res.body,
    (msg) => {
      const type = msg.event as ChatEvent["type"];
      if (!CHAT_EVENTS.has(type)) return;
      let data: unknown;
      try {
        data = JSON.parse(msg.data);
      } catch {
        return;
      }
      onEvent({ type, data } as ChatEvent);
    },
    signal,
  );
}

/**
 * Subscribe to ingestion progress with EventSource. Returns a cleanup function.
 * `onDisconnect` fires when the connection drops before a `done` event.
 */
export function subscribeIngest(
  videoId: string,
  handlers: {
    onEvent: (event: IngestEvent) => void;
    onOpen?: () => void;
    onDisconnect?: () => void;
  },
): () => void {
  const source = new EventSource(`${BASE}/videos/${enc(videoId)}/events`);
  let finished = false;

  const parse = (type: IngestEvent["type"]) => (e: MessageEvent<string>) => {
    try {
      handlers.onEvent({ type, data: JSON.parse(e.data) } as IngestEvent);
    } catch {
      /* ignore malformed event */
    }
    if (type === "done") {
      finished = true;
      source.close();
    }
  };

  source.addEventListener("open", () => handlers.onOpen?.());
  source.addEventListener("progress", parse("progress") as EventListener);
  source.addEventListener("done", parse("done") as EventListener);
  source.addEventListener("error", () => {
    if (!finished) handlers.onDisconnect?.();
  });

  return () => {
    finished = true;
    source.close();
  };
}

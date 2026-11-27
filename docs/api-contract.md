# API contract (v0)

Base path: `/api`. Media files are served from `/media/...` (local storage) or pre-signed S3 URLs.
All URLs returned by the API (`playback_url`, `keyframe_url`, `overlay_url`, `thumbnail_url`, `upload_url`)
are either absolute or root-relative (`/media/...`, `/api/...`) and can be used as-is by the browser.
Times are seconds (float) from the start of the video.

## Types

```ts
type VideoStatus = "uploading" | "queued" | "processing" | "ready" | "failed";

interface Video {
  id: string;                 // uuid
  title: string;
  filename: string;
  status: VideoStatus;
  progress: number;           // 0..1, overall ingestion progress
  duration_s: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  has_audio: boolean | null;
  thumbnail_url: string | null;
  playback_url: string | null; // 720p H.264 proxy (null until ready)
  error: string | null;
  created_at: string;          // ISO 8601
}

type StepName = "probe" | "proxy" | "shots" | "keyframes" | "captions" | "transcript" | "embeddings" | "summaries";
type StepStatus = "pending" | "running" | "done" | "skipped" | "failed";

interface IngestStep {
  step: StepName;
  status: StepStatus;
  progress: number;            // 0..1 within the step
  detail: string | null;       // e.g. "12 / 14 clips captioned"
  error: string | null;
}

interface Timeline {
  duration_s: number;
  shots: { id: string; index: number; t_start: number; t_end: number }[];
  clips: {
    id: string; shot_id: string; t_start: number; t_end: number;
    caption: string; objects: string[]; actions: string[]; on_screen_text: string;
    keyframe_url: string;
  }[];
  transcript: { t_start: number; t_end: number; text: string }[];
}

interface AnalysisReport {
  narration: string;
  key_moments: { t_start: number; t_end: number; title: string; description: string }[];
  entities: { name: string; kind: "person" | "object" | "vehicle" | "place" | "other"; first_seen: number; last_seen: number }[];
  suggested_questions: string[];
  generated_at: string;
}

type Tier = "memory" | "vlm" | "gpu" | "plugin";

interface Evidence { t_start: number; t_end: number; keyframe_url: string | null; reason: string }

interface Answer {
  answer: string;
  verdict: "yes" | "no" | "uncertain" | "n/a";
  confidence: number;          // 0..1
  evidence: Evidence[];
  coverage: string;
  tools_used: string[];
}

interface Step {                // one tool call made by the agent
  id: string;
  tool: string;
  tier: Tier;
  label: string;               // human readable, e.g. "Searching for 'red car crosses'"
  args: Record<string, unknown>;
  summary: string | null;      // short result summary (null while running)
  tool_run_id: string | null;
  overlay_url: string | null;  // annotated video, when the tool produced one
  status: "running" | "done" | "failed";
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;             // user text, or the assistant's answer text
  answer: Answer | null;       // assistant only
  steps: Step[];               // assistant only
  created_at: string;
}

interface ToolRun {
  id: string; tool: string; tier: Tier; params: Record<string, unknown>;
  status: "running" | "done" | "failed"; result: unknown;
  overlay_url: string | null; created_at: string;
}
```

## Endpoints

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| POST | `/api/videos` | `{filename, content_type, size}` | `{video: Video, upload_url, upload_method: "PUT", upload_headers: Record<string,string>}` |
| PUT | `upload_url` | raw file bytes | 200 (S3 or local) |
| POST | `/api/videos/{id}/ingest` | – | `Video` (status `queued`) |
| GET | `/api/videos` | – | `Video[]`, newest first |
| GET | `/api/videos/{id}` | – | `Video` |
| DELETE | `/api/videos/{id}` | – | 204 |
| GET | `/api/videos/{id}/steps` | – | `IngestStep[]` |
| GET | `/api/videos/{id}/events` | – | SSE (EventSource), see below |
| GET | `/api/videos/{id}/timeline` | – | `Timeline` (409 if not ready) |
| POST | `/api/videos/{id}/analysis` | `{refresh?: boolean}` | `AnalysisReport` (cached after first run) |
| GET | `/api/videos/{id}/tool-runs` | – | `ToolRun[]`, newest first |
| GET | `/api/tool-runs/{id}` | – | `ToolRun` |
| POST | `/api/videos/{id}/conversations` | – | `{id, video_id, created_at}` |
| GET | `/api/videos/{id}/conversations` | – | `{id, video_id, created_at}[]`, newest first |
| GET | `/api/conversations/{id}/messages` | – | `Message[]` oldest first |
| POST | `/api/conversations/{id}/messages` | `{content}` | SSE stream over the POST response (use `fetch` + stream reader, not EventSource) |

### Ingestion progress SSE (`GET /api/videos/{id}/events`)

- `event: progress` – `data: {video: Video, steps: IngestStep[]}` – sent on every change.
- `event: done` – `data: {video: Video}` – status became `ready` or `failed`; the server then closes the stream.

### Chat SSE (`POST /api/conversations/{id}/messages`)

Each event is `event: <type>\ndata: <json>\n\n`, in this order:

- `user` – `{message: Message}` – the stored user message.
- `plan` – `{text}` – the agent's short plan (may be omitted).
- `tool_call` – `{step: Step}` – status `running`.
- `tool_result` – `{step: Step}` – same id, status `done`/`failed`, with `summary`, `tool_run_id`, `overlay_url`.
- `answer_delta` – `{text}` – answer text chunks (may be omitted; `final` always carries the full text).
- `final` – `{message: Message}` – the stored assistant message with `answer` and `steps`.
- `error` – `{message}` – then the stream ends.

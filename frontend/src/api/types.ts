// Types mirror docs/api-contract.md (v0). Keep this file in lock-step with the contract.
// Times are seconds (float) from the start of the video.

export type VideoStatus = "uploading" | "queued" | "processing" | "ready" | "failed";

export interface Video {
  id: string;
  title: string;
  filename: string;
  status: VideoStatus;
  /** 0..1, overall ingestion progress */
  progress: number;
  duration_s: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  has_audio: boolean | null;
  thumbnail_url: string | null;
  /** 720p H.264 proxy (null until ready) */
  playback_url: string | null;
  error: string | null;
  /** ISO 8601 */
  created_at: string;
}

export type StepName =
  | "probe"
  | "proxy"
  | "shots"
  | "keyframes"
  | "captions"
  | "transcript"
  | "embeddings"
  | "summaries";

export type StepStatus = "pending" | "running" | "done" | "skipped" | "failed";

export interface IngestStep {
  step: StepName;
  status: StepStatus;
  /** 0..1 within the step */
  progress: number;
  detail: string | null;
  error: string | null;
}

export interface Shot {
  id: string;
  index: number;
  t_start: number;
  t_end: number;
}

export interface Clip {
  id: string;
  shot_id: string;
  t_start: number;
  t_end: number;
  caption: string;
  objects: string[];
  actions: string[];
  on_screen_text: string;
  keyframe_url: string;
}

export interface TranscriptSegment {
  t_start: number;
  t_end: number;
  text: string;
}

export interface Timeline {
  duration_s: number;
  shots: Shot[];
  clips: Clip[];
  transcript: TranscriptSegment[];
}

export interface KeyMoment {
  t_start: number;
  t_end: number;
  title: string;
  description: string;
}

export type EntityKind = "person" | "object" | "vehicle" | "place" | "other";

export interface Entity {
  name: string;
  kind: EntityKind;
  first_seen: number;
  last_seen: number;
}

export interface AnalysisReport {
  narration: string;
  key_moments: KeyMoment[];
  entities: Entity[];
  suggested_questions: string[];
  generated_at: string;
}

export type Tier = "memory" | "vlm" | "gpu" | "plugin";

export interface Evidence {
  t_start: number;
  t_end: number;
  keyframe_url: string | null;
  reason: string;
}

export type Verdict = "yes" | "no" | "uncertain" | "n/a";

export interface Answer {
  answer: string;
  verdict: Verdict;
  /** 0..1 */
  confidence: number;
  evidence: Evidence[];
  coverage: string;
  tools_used: string[];
}

export type RunStatus = "running" | "done" | "failed";

/** One tool call made by the agent. */
export interface Step {
  id: string;
  tool: string;
  tier: Tier;
  label: string;
  args: Record<string, unknown>;
  /** null while running */
  summary: string | null;
  tool_run_id: string | null;
  overlay_url: string | null;
  /** a still picture from the tool (e.g. segment_scene); absent on older steps */
  image_url?: string | null;
  /** a video from the tool (e.g. make_highlight_clip); absent on older steps */
  clip_url?: string | null;
  status: RunStatus;
}

/** An object the user clicked on a paused frame (outlined by SAM), to ask about. */
export interface ObjectSelection {
  id: string;
  t: number;
  x: number;
  y: number;
  /** normalised [x0, y0, x1, y1] */
  bbox: [number, number, number, number];
  /** normalised outline points */
  polygon: [number, number][];
  area_pct: number;
  crop_url: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  /** user text, or the assistant's answer text */
  content: string;
  /** assistant only */
  answer: Answer | null;
  /** assistant only */
  steps: Step[];
  /** user only: the object clicked before asking */
  selection?: ObjectSelection | null;
  created_at: string;
}

export interface ToolRun {
  id: string;
  tool: string;
  tier: Tier;
  params: Record<string, unknown>;
  status: RunStatus;
  result: unknown;
  overlay_url: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  video_id: string;
  created_at: string;
}

// ---- Request / response envelopes ----------------------------------------

export interface CreateVideoRequest {
  filename: string;
  content_type: string;
  size: number;
}

export interface CreateVideoResponse {
  video: Video;
  upload_url: string;
  upload_method: "PUT";
  upload_headers: Record<string, string>;
}

export interface AnalysisRequest {
  refresh?: boolean;
}

export interface SendMessageRequest {
  content: string;
}

// ---- SSE payloads ---------------------------------------------------------

/** GET /api/videos/{id}/events */
export type IngestEvent =
  | { type: "progress"; data: { video: Video; steps: IngestStep[] } }
  | { type: "done"; data: { video: Video } };

/** POST /api/conversations/{id}/messages */
export type ChatEvent =
  | { type: "user"; data: { message: Message } }
  | { type: "plan"; data: { text: string } }
  | { type: "tool_call"; data: { step: Step } }
  | { type: "tool_result"; data: { step: Step } }
  | { type: "answer_delta"; data: { text: string } }
  | { type: "final"; data: { message: Message } }
  | { type: "error"; data: { message: string } };

export type ChatEventType = ChatEvent["type"];

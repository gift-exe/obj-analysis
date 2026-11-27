import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, Minus, UploadCloud, WifiOff, X } from "lucide-react";
import { useEffect, useState } from "react";
import { subscribeIngest } from "../../api/client";
import { errorMessage, keys, patchVideoInList, useSteps } from "../../api/queries";
import type { IngestStep, StepName, Video } from "../../api/types";
import { cn, formatBytes, formatPercent } from "../../lib/format";
import { useUploads } from "../../state/uploads";
import { ErrorNote, ProgressBar, Skeleton, Spinner } from "../ui/primitives";

const STEP_ORDER: StepName[] = [
  "probe",
  "proxy",
  "shots",
  "keyframes",
  "captions",
  "transcript",
  "embeddings",
  "summaries",
];

const STEP_COPY: Record<StepName, { title: string; body: string }> = {
  probe: { title: "Read metadata", body: "Duration, frame rate, resolution and audio" },
  proxy: { title: "Make playback copy", body: "720p H.264 proxy for smooth seeking" },
  shots: { title: "Detect shots", body: "Find cuts and scene boundaries" },
  keyframes: { title: "Extract keyframes", body: "Representative frames for each clip" },
  captions: { title: "Caption clips", body: "Objects, actions and on-screen text" },
  transcript: { title: "Transcribe audio", body: "Speech to timestamped text" },
  embeddings: { title: "Build search index", body: "Embed captions and frames for recall" },
  summaries: { title: "Summarise scenes", body: "Shot and whole-video summaries" },
};

/** Always show all eight steps in order, filling gaps as pending. */
function normaliseSteps(steps: IngestStep[] | undefined): IngestStep[] {
  const byName = new Map(steps?.map((s) => [s.step, s]));
  return STEP_ORDER.map(
    (name) => byName.get(name) ?? { step: name, status: "pending", progress: 0, detail: null, error: null },
  );
}

export function ProcessingView({ video }: { video: Video }) {
  const qc = useQueryClient();
  const live = video.status === "queued" || video.status === "processing";
  const [disconnected, setDisconnected] = useState(false);
  const steps = useSteps(video.id, disconnected && live);
  const { byVideoId } = useUploads();
  const upload = byVideoId(video.id);

  useEffect(() => {
    if (!live) return;
    setDisconnected(false);
    const stop = subscribeIngest(video.id, {
      onOpen: () => setDisconnected(false),
      onDisconnect: () => setDisconnected(true),
      onEvent: (ev) => {
        if (ev.type === "progress") {
          qc.setQueryData(keys.steps(video.id), ev.data.steps);
          patchVideoInList(qc, ev.data.video);
        } else {
          patchVideoInList(qc, ev.data.video);
          void qc.invalidateQueries({ queryKey: keys.steps(video.id) });
          void qc.invalidateQueries({ queryKey: keys.videos });
        }
      },
    });
    return stop;
    // Re-subscribe only when the video or its liveness changes.
  }, [video.id, live, qc]);

  const list = normaliseSteps(steps.data);
  const failedStep = list.find((s) => s.status === "failed");
  const doneCount = list.filter((s) => s.status === "done" || s.status === "skipped").length;

  return (
    <div className="mx-auto w-full max-w-[680px] px-6 py-8">
      {video.status === "uploading" ? (
        <UploadStage progress={upload?.progress ?? null} size={upload?.size ?? null} error={upload?.error ?? null} />
      ) : (
        <>
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 className="font-display text-[26px] font-semibold leading-tight tracking-[-0.02em]">
                {video.status === "failed"
                  ? "Processing stopped"
                  : video.status === "queued"
                    ? "Waiting in the queue"
                    : "Building video memory"}
              </h2>
              <p className="mt-1 text-[13.5px] text-muted">
                {video.status === "failed"
                  ? "One of the steps below failed, so this video can't be searched yet."
                  : "Captions, keyframes and a search index are made once. After that, questions are answered from memory."}
              </p>
            </div>
            <div className="shrink-0 text-right">
              <p className="font-display text-[34px] font-semibold leading-none tracking-[-0.03em] tnum">
                {formatPercent(video.progress)}
              </p>
              <p className="mt-1 text-[12px] text-muted tnum">
                {doneCount} of {STEP_ORDER.length} steps
              </p>
            </div>
          </div>

          <SegmentBar steps={list} />

          {video.status === "failed" && (
            <ErrorNote
              className="mt-5"
              title={failedStep ? `${STEP_COPY[failedStep.step].title} failed` : "Ingestion failed"}
              message={
                video.error ??
                failedStep?.error ??
                "The server didn't report a reason. Delete the video from the library and upload it again."
              }
            />
          )}

          {disconnected && live && (
            <p className="mt-4 flex items-center gap-2 text-[12.5px] text-warn">
              <WifiOff className="size-3.5" aria-hidden />
              Live updates disconnected. Checking progress every few seconds instead.
            </p>
          )}

          {steps.isPending && !steps.data ? (
            <div className="mt-6 space-y-3">
              {STEP_ORDER.map((s) => (
                <Skeleton key={s} className="h-12 w-full" />
              ))}
            </div>
          ) : (
            <>
              {steps.isError && !steps.data && (
                <ErrorNote
                  className="mt-5"
                  title="Couldn't load step details"
                  message={errorMessage(steps.error)}
                  onRetry={() => void steps.refetch()}
                />
              )}
              <ol className="mt-6">
                {list.map((s, i) => (
                  <StepItem key={s.step} step={s} index={i} last={i === list.length - 1} />
                ))}
              </ol>
            </>
          )}
        </>
      )}
    </div>
  );
}

function SegmentBar({ steps }: { steps: IngestStep[] }) {
  return (
    <div className="mt-5 grid grid-cols-8 gap-1" aria-hidden>
      {steps.map((s) => (
        <div key={s.step} className="h-1.5 overflow-hidden rounded-full bg-sunken">
          <div
            className={cn(
              "h-full rounded-full transition-[width] duration-500 ease-out",
              s.status === "failed"
                ? "bg-bad"
                : s.status === "skipped"
                  ? "bg-line-strong"
                  : s.status === "running"
                    ? "stripes animate-sweep bg-mark"
                    : "bg-ok",
            )}
            style={{
              width:
                s.status === "done" || s.status === "skipped" || s.status === "failed"
                  ? "100%"
                  : s.status === "running"
                    ? `${Math.max(6, s.progress * 100)}%`
                    : "0%",
            }}
          />
        </div>
      ))}
    </div>
  );
}

function StepItem({ step, index, last }: { step: IngestStep; index: number; last: boolean }) {
  const copy = STEP_COPY[step.step] ?? { title: step.step, body: "" };
  const running = step.status === "running";
  return (
    <li className="relative flex gap-4 pb-1">
      {!last && (
        <span
          className={cn(
            "absolute left-[13px] top-8 bottom-0 w-px",
            step.status === "done" ? "bg-ok/40" : "bg-line",
          )}
          aria-hidden
        />
      )}
      <StepMarker step={step} index={index} />
      <div
        className={cn(
          "mb-2 min-w-0 flex-1 rounded-lg px-3 py-2 transition-colors",
          running && "bg-raised shadow-panel ring-1 ring-line",
        )}
      >
        <div className="flex items-baseline justify-between gap-3">
          <p
            className={cn(
              "text-[14px] font-medium",
              step.status === "pending" ? "text-muted" : "text-ink",
              step.status === "skipped" && "text-muted line-through decoration-line-strong",
            )}
          >
            {copy.title}
          </p>
          <span
            className={cn(
              "shrink-0 text-[12px] tnum",
              step.status === "failed" ? "text-bad" : running ? "text-mark-ink" : "text-muted",
            )}
          >
            {running
              ? formatPercent(step.progress)
              : step.status === "done"
                ? "Done"
                : step.status === "skipped"
                  ? "Skipped"
                  : step.status === "failed"
                    ? "Failed"
                    : "Waiting"}
          </span>
        </div>
        <p className="text-[12.5px] text-muted">{step.detail ?? copy.body}</p>
        {running && <ProgressBar className="mt-2" value={step.progress} tone="mark" label={copy.title} />}
        {step.status === "failed" && step.error && (
          <p className="mt-1.5 flex items-start gap-1.5 text-[12.5px] text-bad">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            <span className="break-words">{step.error}</span>
          </p>
        )}
      </div>
    </li>
  );
}

function StepMarker({ step, index }: { step: IngestStep; index: number }) {
  const base = "relative z-10 mt-1.5 flex size-[27px] shrink-0 items-center justify-center rounded-full text-[12px] font-semibold tnum";
  switch (step.status) {
    case "done":
      return (
        <span className={cn(base, "bg-ok text-white")}>
          <Check className="size-3.5" strokeWidth={3} aria-label="Done" />
        </span>
      );
    case "failed":
      return (
        <span className={cn(base, "bg-bad text-white")}>
          <X className="size-3.5" strokeWidth={3} aria-label="Failed" />
        </span>
      );
    case "skipped":
      return (
        <span className={cn(base, "border border-line bg-surface text-muted")}>
          <Minus className="size-3.5" aria-label="Skipped" />
        </span>
      );
    case "running":
      return (
        <span className={cn(base, "bg-mark-soft text-mark-ink")}>
          <Spinner className="absolute inset-0 size-[27px] text-mark" />
          {index + 1}
        </span>
      );
    default:
      return <span className={cn(base, "border border-line bg-surface text-muted")}>{index + 1}</span>;
  }
}

function UploadStage({ progress, size, error }: { progress: number | null; size: number | null; error: string | null }) {
  return (
    <div className="flex flex-col items-center py-12 text-center">
      <div className="mb-4 flex size-14 items-center justify-center rounded-2xl bg-accent-soft text-accent">
        <UploadCloud className="size-7" strokeWidth={1.5} aria-hidden />
      </div>
      <h2 className="font-display text-[24px] font-semibold tracking-[-0.02em]">
        {error ? "Upload failed" : "Uploading"}
      </h2>
      {error ? (
        <ErrorNote className="mt-4 w-full max-w-sm text-left" message={error} />
      ) : progress == null ? (
        <p className="mt-2 max-w-sm text-[13.5px] text-muted">
          Waiting for the upload to finish. If it was interrupted, delete this video from the library and upload it
          again.
        </p>
      ) : (
        <>
          <p className="mt-1 font-display text-[32px] font-semibold tracking-[-0.03em] tnum">
            {formatPercent(progress)}
          </p>
          <ProgressBar className="mt-3 w-full max-w-sm" value={progress} label="Upload progress" />
          {size != null && (
            <p className="mt-2 text-[12.5px] text-muted tnum">
              {formatBytes(size * progress)} of {formatBytes(size)}
            </p>
          )}
          <p className="mt-4 max-w-sm text-[13px] text-muted">
            Keep this tab open. Processing starts as soon as the upload finishes.
          </p>
        </>
      )}
    </div>
  );
}

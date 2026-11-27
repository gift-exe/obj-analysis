import { useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";
import { errorMessage, useTimeline } from "../../api/queries";
import type { Clip, Timeline as TimelineData, Video } from "../../api/types";
import { clamp, cn, formatRange, formatTime } from "../../lib/format";
import { usePlayer, usePlayerClock } from "../../state/player";
import { ErrorNote, Skeleton } from "../ui/primitives";

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

function tickStep(duration: number, maxTicks = 10) {
  return TICK_STEPS.find((s) => duration / s <= maxTicks) ?? 3600;
}

const pct = (t: number, d: number) => `${(clamp(t, 0, d) / d) * 100}%`;

export function TimelineStrip({ video }: { video: Video }) {
  const timeline = useTimeline(video.id, video.status === "ready");

  if (timeline.isPending) {
    return (
      <div className="space-y-2" aria-busy="true" aria-label="Loading timeline">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-3 w-full" />
      </div>
    );
  }
  if (timeline.isError) {
    return (
      <ErrorNote
        title="Couldn't load the timeline"
        message={errorMessage(timeline.error)}
        onRetry={() => void timeline.refetch()}
      />
    );
  }
  return <Strip data={timeline.data} fallbackDuration={video.duration_s} />;
}

interface Hover {
  x: number;
  t: number;
  clip: Clip | null;
}

function Strip({ data, fallbackDuration }: { data: TimelineData; fallbackDuration: number | null }) {
  const { seek, evidence, keyMoments, highlight } = usePlayer();
  const { currentTime } = usePlayerClock();
  const trackRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  const duration = data.duration_s > 0 ? data.duration_s : fallbackDuration ?? 0;

  const clipsByShot = useMemo(() => {
    const m = new Map<string, Clip[]>();
    for (const c of data.clips) {
      const arr = m.get(c.shot_id);
      if (arr) arr.push(c);
      else m.set(c.shot_id, [c]);
    }
    return m;
  }, [data.clips]);

  const ticks = useMemo(() => {
    if (duration <= 0) return [];
    const step = tickStep(duration);
    const out: number[] = [];
    for (let t = 0; t <= duration + 1e-6; t += step) out.push(t);
    return out;
  }, [duration]);

  if (duration <= 0) {
    return <p className="text-[13px] text-muted">This video has no duration, so there's no timeline to show.</p>;
  }

  const timeAt = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return { t: 0, x: 0 };
    const x = clamp(clientX - rect.left, 0, rect.width);
    return { t: (x / rect.width) * duration, x };
  };

  const clipAt = (t: number) =>
    data.clips.find((c) => t >= c.t_start && t < c.t_end) ??
    data.clips.find((c) => Math.abs(c.t_end - t) < 0.05) ??
    null;

  const onMove = (e: MouseEvent) => {
    const { t, x } = timeAt(e.clientX);
    setHover({ t, x, clip: clipAt(t) });
  };

  const onClick = (e: MouseEvent) => seek(timeAt(e.clientX).t);

  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 1 : 5;
    let next: number | null = null;
    if (e.key === "ArrowLeft") next = currentTime - step;
    else if (e.key === "ArrowRight") next = currentTime + step;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = duration;
    if (next !== null) {
      e.preventDefault();
      seek(clamp(next, 0, duration));
    }
  };

  const transcriptAt = (t: number) => data.transcript.find((s) => t >= s.t_start && t < s.t_end) ?? null;
  const hasMarkers = evidence.length > 0 || keyMoments.length > 0;

  return (
    <section aria-label="Timeline" className="select-none">
      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label="Seek"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        aria-valuetext={formatTime(currentTime)}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={onClick}
        onKeyDown={onKey}
        className="relative cursor-pointer rounded-lg focus-visible:outline-offset-4"
      >
        {/* Marker lane: key moments (cobalt) and evidence (yellow). */}
        <div className="relative h-5">
          {keyMoments.map((k, i) => (
            <span
              key={`k${i}`}
              title={`${k.title} (${formatRange(k.t_start, k.t_end)})`}
              className="absolute top-1 h-2 rounded-full border border-accent/60 bg-accent-soft"
              style={{
                left: pct(k.t_start, duration),
                width: `max(6px, ${((clamp(k.t_end, 0, duration) - clamp(k.t_start, 0, duration)) / duration) * 100}%)`,
              }}
            />
          ))}
          {evidence.map((ev, i) => (
            <span
              key={`e${i}`}
              title={`Evidence ${formatRange(ev.t_start, ev.t_end)}: ${ev.reason}`}
              className="absolute bottom-0 flex h-4 -translate-x-[1px] items-end"
              style={{
                left: pct(ev.t_start, duration),
                width: `max(3px, ${((clamp(ev.t_end, 0, duration) - clamp(ev.t_start, 0, duration)) / duration) * 100}%)`,
              }}
            >
              <span className="h-full w-full rounded-t-[3px] border-x-2 border-t-2 border-mark bg-mark/25" />
            </span>
          ))}
        </div>

        {/* Film lane: shots with keyframes. */}
        <div className="relative h-16 overflow-hidden rounded-md bg-film ring-1 ring-black/10">
          {data.shots.length === 0 && data.clips.length === 0 ? (
            <p className="flex h-full items-center justify-center text-[12px] text-white/50">No shots detected</p>
          ) : data.shots.length === 0 ? (
            data.clips.map((c) => <ClipThumb key={c.id} clip={c} left={pct(c.t_start, duration)} width={((c.t_end - c.t_start) / duration) * 100} />)
          ) : (
            data.shots.map((s) => {
              const len = Math.max(0.001, s.t_end - s.t_start);
              const clips = clipsByShot.get(s.id) ?? [];
              return (
                <div
                  key={s.id}
                  className="absolute inset-y-0 overflow-hidden border-x border-film bg-white/[0.04]"
                  style={{ left: pct(s.t_start, duration), width: `${(len / duration) * 100}%` }}
                >
                  {clips.map((c) => (
                    <ClipThumb
                      key={c.id}
                      clip={c}
                      left={`${((c.t_start - s.t_start) / len) * 100}%`}
                      width={((c.t_end - c.t_start) / len) * 100}
                    />
                  ))}
                  <span className="pointer-events-none absolute left-1 top-1 rounded-[3px] bg-black/55 px-1 text-[10px] font-medium leading-4 text-white/85 tnum">
                    {s.index + 1}
                  </span>
                </div>
              );
            })
          )}
          {hover?.clip && (
            <span
              className="pointer-events-none absolute inset-y-0 ring-2 ring-inset ring-white/80"
              style={{
                left: pct(hover.clip.t_start, duration),
                width: `${((hover.clip.t_end - hover.clip.t_start) / duration) * 100}%`,
              }}
            />
          )}
        </div>

        {/* Ruler */}
        <div className="relative mt-1 h-4">
          {ticks.map((t) => (
            <span
              key={t}
              className="absolute top-0 -translate-x-1/2 text-[10.5px] text-muted tnum first:translate-x-0"
              style={{ left: pct(t, duration) }}
            >
              {formatTime(t)}
            </span>
          ))}
        </div>

        {/* Highlighted range (from an evidence card or key moment). */}
        {highlight && (
          <span
            className="pointer-events-none absolute top-0 bottom-4 rounded-sm bg-mark/20 ring-1 ring-mark/60"
            style={{
              left: pct(highlight.t_start, duration),
              width: `max(3px, ${((clamp(highlight.t_end, 0, duration) - clamp(highlight.t_start, 0, duration)) / duration) * 100}%)`,
            }}
          />
        )}

        {/* Hover line */}
        {hover && (
          <span
            className="pointer-events-none absolute top-4 bottom-4 w-px bg-ink/40"
            style={{ left: hover.x }}
            aria-hidden
          />
        )}

        {/* Playhead */}
        <span
          className="pointer-events-none absolute top-0 bottom-4 w-0.5 -translate-x-1/2 bg-mark shadow-[0_0_0_1px_rgb(0_0_0/0.25)]"
          style={{ left: pct(currentTime, duration) }}
          aria-hidden
        >
          <span className="absolute -top-0.5 left-1/2 size-2.5 -translate-x-1/2 rotate-45 rounded-[2px] bg-mark" />
        </span>

        {hover && (
          <HoverCard
            hover={hover}
            width={trackRef.current?.clientWidth ?? 0}
            transcript={transcriptAt(hover.t)?.text ?? null}
          />
        )}
      </div>

      {hasMarkers && (
        <div className="mt-1.5 flex items-center gap-4 text-[11.5px] text-muted">
          {evidence.length > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-1.5 rounded-t-[2px] border-x-2 border-t-2 border-mark bg-mark/25" aria-hidden />
              Evidence from the latest answer
            </span>
          )}
          {keyMoments.length > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2 w-4 rounded-full border border-accent/60 bg-accent-soft" aria-hidden />
              Key moments
            </span>
          )}
        </div>
      )}
    </section>
  );
}

function ClipThumb({ clip, left, width }: { clip: Clip; left: string; width: number }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="absolute inset-y-0 overflow-hidden border-r border-black/40" style={{ left, width: `${width}%` }}>
      {clip.keyframe_url && !failed ? (
        <img
          src={clip.keyframe_url}
          alt=""
          loading="lazy"
          draggable={false}
          onError={() => setFailed(true)}
          className="size-full object-cover opacity-85"
        />
      ) : (
        <div className="size-full bg-white/[0.06]" />
      )}
    </div>
  );
}

function HoverCard({ hover, width, transcript }: { hover: Hover; width: number; transcript: string | null }) {
  const CARD = 300;
  const left = clamp(hover.x - CARD / 2, 0, Math.max(0, width - CARD));
  const clip = hover.clip;
  return (
    <div
      className="pointer-events-none absolute bottom-[calc(100%+8px)] z-30 rounded-xl border border-line bg-raised p-2.5 text-left shadow-pop animate-rise"
      style={{ left, width: CARD }}
      role="tooltip"
    >
      {clip ? (
        <div className="flex gap-2.5">
          {clip.keyframe_url && (
            <img src={clip.keyframe_url} alt="" className="h-[60px] w-[96px] shrink-0 rounded-md bg-film object-cover" />
          )}
          <div className="min-w-0 flex-1">
            <p className="text-[11.5px] font-semibold text-mark-ink tnum">
              {formatRange(clip.t_start, clip.t_end)}
              <span className="font-normal text-muted"> at {formatTime(hover.t)}</span>
            </p>
            <p className="mt-0.5 line-clamp-3 text-[12.5px] leading-snug text-ink">
              {clip.caption || "No caption for this clip."}
            </p>
          </div>
        </div>
      ) : (
        <p className="text-[12px] text-muted tnum">{formatTime(hover.t)}</p>
      )}
      {clip && clip.objects.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {clip.objects.slice(0, 10).map((o, i) => (
            <span key={`${o}${i}`} className="rounded-[4px] bg-sunken px-1.5 py-px text-[11px] text-ink-2">
              {o}
            </span>
          ))}
          {clip.objects.length > 10 && (
            <span className="px-1 text-[11px] text-muted">+{clip.objects.length - 10}</span>
          )}
        </div>
      )}
      {clip && clip.actions.length > 0 && (
        <p className="mt-1.5 text-[11.5px] text-muted">
          <span className="text-ink-2">Actions:</span> {clip.actions.join(", ")}
        </p>
      )}
      {clip?.on_screen_text && (
        <p className="mt-1.5 text-[11.5px] text-muted">
          <span className="text-ink-2">On screen:</span> “{clip.on_screen_text}”
        </p>
      )}
      {transcript && (
        <p className={cn("mt-1.5 border-l-2 border-line-strong pl-2 text-[11.5px] italic text-ink-2", !clip && "mt-1")}>
          {transcript}
        </p>
      )}
    </div>
  );
}

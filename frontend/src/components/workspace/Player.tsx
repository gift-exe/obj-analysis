import { Layers, Maximize2, MousePointerClick, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type MouseEvent } from "react";
import { useToolRuns } from "../../api/queries";
import type { Video } from "../../api/types";
import { cn, formatTime } from "../../lib/format";
import { useObjectSelection } from "../../state/objectSelection";
import { usePlayer, usePlayerClock } from "../../state/player";
import { IconButton, Spinner } from "../ui/primitives";

/** Restore a playback position once the element can accept it. */
function handOver(el: HTMLVideoElement, t: number, play: boolean) {
  const apply = () => {
    const max = Number.isFinite(el.duration) ? Math.max(0, el.duration - 0.05) : t;
    el.currentTime = Math.min(t, max);
    if (play) void el.play().catch(() => undefined);
  };
  if (el.readyState >= 1) {
    apply();
    return;
  }
  el.addEventListener("loadedmetadata", apply, { once: true });
  // An element left at preload="none" may sit idle; kick off loading.
  if (el.networkState === HTMLMediaElement.NETWORK_IDLE || el.networkState === HTMLMediaElement.NETWORK_EMPTY) el.load();
}

/** Where an object-contain video's picture sits inside its box (letterboxing), in px. */
function contentBox(boxW: number, boxH: number, vw: number, vh: number) {
  if (!boxW || !boxH || !vw || !vh) return { left: 0, top: 0, width: boxW, height: boxH };
  const scale = Math.min(boxW / vw, boxH / vh);
  const width = vw * scale;
  const height = vh * scale;
  return { left: (boxW - width) / 2, top: (boxH - height) / 2, width, height };
}

export function Player({ video }: { video: Video }) {
  const { clock, registerMedia, overlayOn, overlayUrl, setOverlayOn, setFallbackOverlay, togglePlay, seek } =
    usePlayer();
  const { currentTime, duration, playing } = usePlayerClock();
  const toolRuns = useToolRuns(video.id, true);
  const objSel = useObjectSelection();
  const [box, setBox] = useState({ w: 0, h: 0 });

  const frameRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLVideoElement>(null);
  const [muted, setMuted] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);

  // Default overlay: the newest tool run that produced one.
  useEffect(() => {
    const latest = toolRuns.data?.find((r) => r.overlay_url)?.overlay_url ?? null;
    setFallbackOverlay(latest);
  }, [toolRuns.data, setFallbackOverlay]);

  const overlayActive = overlayOn && !!overlayUrl;

  // Wire the visible element to the shared clock; hand position over on switch.
  useEffect(() => {
    const active = overlayActive ? overlayRef.current : mainRef.current;
    const hidden = overlayActive ? mainRef.current : overlayRef.current;
    if (!active) return;

    const { currentTime: t, playing: wasPlaying } = clock.get();
    if (hidden && !hidden.paused) hidden.pause();
    if (Math.abs(active.currentTime - t) > 0.05 || active.readyState < 1) handOver(active, t, wasPlaying);
    else if (wasPlaying && active.paused) void active.play().catch(() => undefined);
    registerMedia(active);

    let raf = 0;
    const tick = () => {
      clock.set({ currentTime: active.currentTime });
      raf = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      clock.set({ playing: true });
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const onPause = () => {
      cancelAnimationFrame(raf);
      clock.set({ playing: false, currentTime: active.currentTime });
    };
    const onTime = () => {
      if (active.paused) clock.set({ currentTime: active.currentTime });
    };
    const onDuration = () => {
      if (Number.isFinite(active.duration)) clock.set({ duration: active.duration });
    };
    const onWaiting = () => setBuffering(true);
    const onReady = () => setBuffering(false);

    active.addEventListener("play", onPlay);
    active.addEventListener("pause", onPause);
    active.addEventListener("ended", onPause);
    active.addEventListener("timeupdate", onTime);
    active.addEventListener("seeked", onTime);
    active.addEventListener("durationchange", onDuration);
    active.addEventListener("loadedmetadata", onDuration);
    active.addEventListener("waiting", onWaiting);
    active.addEventListener("playing", onReady);
    active.addEventListener("canplay", onReady);
    onDuration();
    if (!active.paused) onPlay();

    return () => {
      cancelAnimationFrame(raf);
      active.removeEventListener("play", onPlay);
      active.removeEventListener("pause", onPause);
      active.removeEventListener("ended", onPause);
      active.removeEventListener("timeupdate", onTime);
      active.removeEventListener("seeked", onTime);
      active.removeEventListener("durationchange", onDuration);
      active.removeEventListener("loadedmetadata", onDuration);
      active.removeEventListener("waiting", onWaiting);
      active.removeEventListener("playing", onReady);
      active.removeEventListener("canplay", onReady);
      registerMedia(null);
    };
  }, [overlayActive, overlayUrl, clock, registerMedia]);

  useEffect(() => setOverlayError(null), [overlayUrl]);

  // Track the frame size so the selection outline can follow the letterboxed picture.
  useLayoutEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => e && setBox({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const vw = mainRef.current?.videoWidth || video.width || 16;
  const vh = mainRef.current?.videoHeight || video.height || 9;
  const pic = contentBox(box.w, box.h, vw, vh);
  const sel = objSel.selection;
  const selVisible = !!sel && Math.abs(currentTime - sel.t) < 0.25;

  // Entering select mode pauses, so the click lands on the frame the user is looking at.
  useEffect(() => {
    if (objSel.selectMode && playing) togglePlay();
  }, [objSel.selectMode, playing, togglePlay]);

  const onSelectClick = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - r.left - pic.left) / pic.width;
    const y = (e.clientY - r.top - pic.top) / pic.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return; // clicked the letterbox bars
    void objSel.selectAt(clock.get().currentTime, x, y);
  };

  const toggleFullscreen = () => {
    const el = frameRef.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen?.().catch(() => undefined);
  };

  const shownDuration = duration || video.duration_s || 0;
  const overlayLoading = overlayActive && buffering;

  return (
    <section aria-label="Video player" className="flex min-h-0 flex-col">
      <div
        ref={frameRef}
        className="group relative aspect-video w-full overflow-hidden rounded-xl bg-film shadow-panel"
      >
        <video
          ref={mainRef}
          src={video.playback_url ?? undefined}
          poster={video.thumbnail_url ?? undefined}
          preload="auto"
          playsInline
          muted={muted}
          onClick={togglePlay}
          className={cn(
            "absolute inset-0 size-full object-contain transition-opacity duration-200",
            overlayActive ? "pointer-events-none opacity-0" : "opacity-100",
          )}
        />
        {overlayUrl && (
          <video
            ref={overlayRef}
            src={overlayUrl}
            // Don't compete with the main video for bandwidth until it's shown.
            preload={overlayActive ? "auto" : "none"}
            playsInline
            muted={muted}
            onClick={togglePlay}
            onError={() => {
              setOverlayError("The overlay video couldn't be loaded.");
              setOverlayOn(false);
            }}
            className={cn(
              "absolute inset-0 size-full object-contain transition-opacity duration-200",
              overlayActive ? "opacity-100" : "pointer-events-none opacity-0",
            )}
          />
        )}

        {overlayActive && (
          <span className="pointer-events-none absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 text-[11.5px] font-medium text-white backdrop-blur-sm animate-rise">
            <span className="size-1.5 rounded-full bg-tier-gpu" aria-hidden />
            Analysis overlay
          </span>
        )}
        {selVisible && sel && (
          <svg
            className="pointer-events-none absolute animate-rise"
            style={{ left: pic.left, top: pic.top, width: pic.width, height: pic.height }}
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            aria-label="Selected object"
          >
            <polygon
              points={sel.polygon.map(([x, y]) => `${x},${y}`).join(" ")}
              fill="rgba(0, 230, 118, 0.22)"
              stroke="#00e676"
              strokeWidth={2.5}
              vectorEffect="non-scaling-stroke"
              strokeLinejoin="round"
            />
          </svg>
        )}
        {sel && !selVisible && !objSel.selectMode && (
          <button
            type="button"
            onClick={() => seek(sel.t)}
            className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 text-[11.5px] font-medium text-white backdrop-blur-sm hover:bg-black/75"
          >
            <span className="size-1.5 rounded-full bg-[#00e676]" aria-hidden />
            Selected object at {formatTime(sel.t)}: show
          </button>
        )}
        {objSel.selectMode && (
          <div
            role="button"
            tabIndex={0}
            aria-label="Click an object to select it"
            onClick={onSelectClick}
            onKeyDown={(e) => e.key === "Escape" && objSel.setSelectMode(false)}
            className="absolute inset-0 z-10 cursor-crosshair"
          >
            <span className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 rounded-md bg-black/65 px-2.5 py-1 text-[12px] font-medium text-white backdrop-blur-sm">
              {objSel.busy ? "Outlining…" : "Click an object to ask about it · Esc to cancel"}
            </span>
          </div>
        )}
        {objSel.error && (
          <span className="absolute bottom-3 right-3 z-20 max-w-[60%] rounded-md bg-black/70 px-2 py-1 text-[11.5px] text-white">
            {objSel.error}
          </span>
        )}
        {(overlayLoading || buffering) && (
          <span className="pointer-events-none absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 text-[11.5px] text-white">
            <Spinner className="size-3" />
            {overlayLoading ? "Loading overlay" : "Buffering"}
          </span>
        )}
        {!playing && !objSel.selectMode && (
          <button
            type="button"
            onClick={togglePlay}
            aria-label="Play"
            className="absolute left-1/2 top-1/2 flex size-14 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-black/55 text-white opacity-90 backdrop-blur-sm transition hover:scale-105 hover:opacity-100"
          >
            <Play className="ml-0.5 size-6 fill-current" />
          </button>
        )}
      </div>

      <div className="mt-2 flex items-center gap-1">
        <IconButton label={playing ? "Pause" : "Play"} onClick={togglePlay} className="text-ink">
          {playing ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
        </IconButton>
        <span className="min-w-[88px] px-1 text-[13px] text-ink-2 tnum">
          <span className="text-ink">{formatTime(currentTime)}</span>
          <span className="text-muted"> / {formatTime(shownDuration)}</span>
        </span>
        <IconButton label={muted ? "Unmute" : "Mute"} onClick={() => setMuted((m) => !m)}>
          {muted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
        </IconButton>

        <div className="ml-auto flex items-center gap-1">
          {overlayError && <span className="mr-2 text-[12px] text-bad">{overlayError}</span>}
          <button
            type="button"
            aria-pressed={objSel.selectMode}
            title="Pause and click an object in the video to ask about it"
            onClick={() => objSel.setSelectMode(!objSel.selectMode)}
            className={cn(
              "inline-flex h-8 items-center gap-2 rounded-lg px-2.5 text-[13px] font-medium transition-colors",
              objSel.selectMode ? "bg-accent text-accent-ink" : "text-ink-2 hover:bg-sunken",
            )}
          >
            <MousePointerClick className="size-4" aria-hidden />
            {objSel.busy ? "Outlining…" : objSel.selectMode ? "Click the object" : "Select object"}
          </button>
          <OverlaySwitch
            on={overlayActive}
            disabled={!overlayUrl}
            onChange={(on) => setOverlayOn(on)}
            hint={
              overlayUrl
                ? "Play the annotated video with boxes, masks and track IDs"
                : toolRuns.isPending
                  ? "Checking for overlays…"
                  : "No overlay yet. Ask a question that needs object tracking to make one."
            }
          />
          <IconButton label="Full screen" onClick={toggleFullscreen}>
            <Maximize2 className="size-4" />
          </IconButton>
        </div>
      </div>
    </section>
  );
}

function OverlaySwitch({
  on,
  disabled,
  onChange,
  hint,
}: {
  on: boolean;
  disabled: boolean;
  onChange: (on: boolean) => void;
  hint: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      title={hint}
      onClick={() => onChange(!on)}
      className={cn(
        "inline-flex h-8 items-center gap-2 rounded-lg px-2.5 text-[13px] font-medium transition-colors",
        disabled ? "cursor-not-allowed text-muted opacity-60" : on ? "bg-tier-gpu-soft text-tier-gpu" : "text-ink-2 hover:bg-sunken",
      )}
    >
      <Layers className="size-4" aria-hidden />
      Show analysis overlay
      <span
        className={cn(
          "relative h-[18px] w-[30px] rounded-full transition-colors",
          on ? "bg-tier-gpu" : "bg-line-strong",
        )}
        aria-hidden
      >
        <span
          className={cn(
            "absolute top-[2px] size-[14px] rounded-full bg-white shadow transition-[left] duration-150",
            on ? "left-[14px]" : "left-[2px]",
          )}
        />
      </span>
    </button>
  );
}

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import type { Evidence, KeyMoment } from "../api/types";

/**
 * Shared player state so chat, timeline and player can coordinate.
 *
 * Split in two: `PlayerContext` holds slow-changing state (overlay, markers)
 * and actions; the clock (currentTime / duration / playing) lives in a tiny
 * external store so only components that render time re-render on each tick.
 */

export interface Clock {
  currentTime: number;
  duration: number;
  playing: boolean;
}

function createClock() {
  let state: Clock = { currentTime: 0, duration: 0, playing: false };
  const listeners = new Set<() => void>();
  return {
    get: () => state,
    set(patch: Partial<Clock>) {
      const next = { ...state, ...patch };
      if (
        next.currentTime === state.currentTime &&
        next.duration === state.duration &&
        next.playing === state.playing
      )
        return;
      state = next;
      listeners.forEach((l) => l());
    },
    subscribe(l: () => void) {
      listeners.add(l);
      return () => listeners.delete(l);
    },
  };
}

type ClockStore = ReturnType<typeof createClock>;

export interface TimeRange {
  t_start: number;
  t_end: number;
}

interface PlayerContextValue {
  clock: ClockStore;
  /** Seek the active video. Optionally highlight a range on the timeline. */
  seek: (t: number, opts?: { play?: boolean; range?: TimeRange }) => void;
  togglePlay: () => void;
  /** The player registers whichever <video> is currently visible. */
  registerMedia: (el: HTMLVideoElement | null) => void;

  overlayOn: boolean;
  setOverlayOn: (on: boolean) => void;
  /** Effective overlay: the one picked from chat, else the latest tool run. */
  overlayUrl: string | null;
  pickedOverlayUrl: string | null;
  /** Pick an overlay (e.g. from a chat step) and switch it on. */
  showOverlay: (url: string) => void;
  setFallbackOverlay: (url: string | null) => void;

  evidence: Evidence[];
  setEvidence: (e: Evidence[]) => void;
  keyMoments: KeyMoment[];
  setKeyMoments: (k: KeyMoment[]) => void;
  highlight: TimeRange | null;
}

const PlayerContext = createContext<PlayerContextValue | null>(null);

export function PlayerProvider({ children }: { children: ReactNode }) {
  const clockRef = useRef<ClockStore | null>(null);
  if (!clockRef.current) clockRef.current = createClock();
  const clock = clockRef.current;

  const mediaRef = useRef<HTMLVideoElement | null>(null);
  const pendingSeek = useRef<{ t: number; play: boolean } | null>(null);

  const [overlayOn, setOverlayOnState] = useState(false);
  const [pickedOverlayUrl, setPicked] = useState<string | null>(null);
  const [fallbackOverlay, setFallbackOverlay] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [keyMoments, setKeyMoments] = useState<KeyMoment[]>([]);
  const [highlight, setHighlight] = useState<TimeRange | null>(null);

  const overlayUrl = pickedOverlayUrl ?? fallbackOverlay;

  const applySeek = useCallback(
    (el: HTMLVideoElement, t: number, play: boolean) => {
      const max = Number.isFinite(el.duration) ? el.duration : Infinity;
      const target = Math.max(0, Math.min(t, max));
      el.currentTime = target;
      clock.set({ currentTime: target });
      if (play) void el.play().catch(() => undefined);
    },
    [clock],
  );

  const registerMedia = useCallback(
    (el: HTMLVideoElement | null) => {
      mediaRef.current = el;
      if (el && pendingSeek.current) {
        const { t, play } = pendingSeek.current;
        pendingSeek.current = null;
        applySeek(el, t, play);
      }
    },
    [applySeek],
  );

  const seek = useCallback<PlayerContextValue["seek"]>(
    (t, opts) => {
      setHighlight(opts?.range ?? null);
      const el = mediaRef.current;
      if (!el || el.readyState === 0) {
        pendingSeek.current = { t, play: !!opts?.play };
        clock.set({ currentTime: t });
        return;
      }
      applySeek(el, t, !!opts?.play);
    },
    [applySeek, clock],
  );

  const togglePlay = useCallback(() => {
    const el = mediaRef.current;
    if (!el) return;
    if (el.paused) void el.play().catch(() => undefined);
    else el.pause();
  }, []);

  const setOverlayOn = useCallback((on: boolean) => setOverlayOnState(on), []);
  const showOverlay = useCallback((url: string) => {
    setPicked(url);
    setOverlayOnState(true);
  }, []);

  const value = useMemo<PlayerContextValue>(
    () => ({
      clock,
      seek,
      togglePlay,
      registerMedia,
      overlayOn: overlayOn && overlayUrl !== null,
      setOverlayOn,
      overlayUrl,
      pickedOverlayUrl,
      showOverlay,
      setFallbackOverlay,
      evidence,
      setEvidence,
      keyMoments,
      setKeyMoments,
      highlight,
    }),
    [
      clock,
      seek,
      togglePlay,
      registerMedia,
      overlayOn,
      setOverlayOn,
      overlayUrl,
      pickedOverlayUrl,
      showOverlay,
      evidence,
      keyMoments,
      highlight,
    ],
  );

  return <PlayerContext.Provider value={value}>{children}</PlayerContext.Provider>;
}

export function usePlayer(): PlayerContextValue {
  const ctx = useContext(PlayerContext);
  if (!ctx) throw new Error("usePlayer must be used inside <PlayerProvider>");
  return ctx;
}

/** Subscribe to the playback clock (re-renders on every time update). */
export function usePlayerClock(): Clock {
  const { clock } = usePlayer();
  return useSyncExternalStore(clock.subscribe, clock.get, clock.get);
}

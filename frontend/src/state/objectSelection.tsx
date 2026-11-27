import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../api/client";
import { errorMessage } from "../api/queries";
import type { ObjectSelection } from "../api/types";

/**
 * Click-to-ask. "Select mode" turns a click on the paused video into a SAM outline of the object under it;
 * the selection then rides along with chat messages until cleared. Lives inside the per-video PlayerProvider,
 * so switching videos clears it.
 */
interface ObjectSelectionValue {
  selectMode: boolean;
  setSelectMode: (on: boolean) => void;
  selection: ObjectSelection | null;
  /** outlining in progress */
  busy: boolean;
  error: string | null;
  selectAt: (t: number, x: number, y: number) => Promise<void>;
  clear: () => void;
}

const Ctx = createContext<ObjectSelectionValue | null>(null);

export function ObjectSelectionProvider({ videoId, children }: { videoId: string; children: ReactNode }) {
  const [selectMode, setSelectModeState] = useState(false);
  const [selection, setSelection] = useState<ObjectSelection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  const setSelectMode = useCallback((on: boolean) => {
    setSelectModeState(on);
    setError(null);
  }, []);

  const selectAt = useCallback(
    async (t: number, x: number, y: number) => {
      const mine = ++seq.current;
      setBusy(true);
      setError(null);
      try {
        const sel = await api.selectObject(videoId, t, x, y);
        if (mine !== seq.current) return; // a newer click won
        setSelection(sel);
        setSelectModeState(false);
      } catch (err) {
        if (mine === seq.current) setError(errorMessage(err));
      } finally {
        if (mine === seq.current) setBusy(false);
      }
    },
    [videoId],
  );

  const clear = useCallback(() => {
    seq.current++;
    setSelection(null);
    setBusy(false);
    setError(null);
  }, []);

  const value = useMemo(
    () => ({ selectMode, setSelectMode, selection, busy, error, selectAt, clear }),
    [selectMode, setSelectMode, selection, busy, error, selectAt, clear],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useObjectSelection(): ObjectSelectionValue {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useObjectSelection must be used inside <ObjectSelectionProvider>");
  return ctx;
}

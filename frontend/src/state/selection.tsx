import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

/** Selected video id, mirrored to the URL hash (#/v/<id>) so reloads keep it. */

const PREFIX = "#/v/";

function readHash(): string | null {
  const h = window.location.hash;
  return h.startsWith(PREFIX) ? decodeURIComponent(h.slice(PREFIX.length)) || null : null;
}

const SelectionContext = createContext<{
  selectedId: string | null;
  select: (id: string | null) => void;
} | null>(null);

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [selectedId, setSelectedId] = useState<string | null>(readHash);

  useEffect(() => {
    const onHash = () => setSelectedId(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const select = useCallback((id: string | null) => {
    setSelectedId(id);
    const base = window.location.pathname + window.location.search;
    window.history.replaceState(null, "", id ? `${base}${PREFIX}${encodeURIComponent(id)}` : base);
  }, []);

  return <SelectionContext.Provider value={{ selectedId, select }}>{children}</SelectionContext.Provider>;
}

export function useSelection() {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error("useSelection must be used inside <SelectionProvider>");
  return ctx;
}

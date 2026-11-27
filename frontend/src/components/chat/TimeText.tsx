import type { ReactNode } from "react";
import { parseTimestamp, TIMESTAMP_RE } from "../../lib/format";
import { usePlayer } from "../../state/player";

/** Renders text with "m:ss" timestamps turned into seek buttons. */
export function TimeText({ text }: { text: string }) {
  const { seek } = usePlayer();
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TIMESTAMP_RE)) {
    const idx = m.index ?? 0;
    if (idx > last) parts.push(text.slice(last, idx));
    const t = parseTimestamp(m[1] ?? "0", m[2] ?? "0");
    parts.push(
      <button
        key={`${idx}-${m[0]}`}
        type="button"
        onClick={() => seek(t)}
        className="rounded-[4px] bg-mark-soft px-1 font-medium text-mark-ink tnum hover:bg-mark hover:text-black"
      >
        {m[0]}
      </button>,
    );
    last = idx + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}

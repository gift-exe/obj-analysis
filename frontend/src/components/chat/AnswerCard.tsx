import { ImageOff, ScanSearch, Wrench } from "lucide-react";
import { useState } from "react";
import type { Answer, Evidence } from "../../api/types";
import { cn, formatPercent, formatRange } from "../../lib/format";
import { usePlayer } from "../../state/player";
import { VerdictChip } from "../ui/badges";
import { TimeText } from "./TimeText";

function confidenceTone(c: number) {
  if (c >= 0.75) return "bg-ok";
  if (c >= 0.45) return "bg-warn";
  return "bg-bad";
}

export function AnswerCard({ answer, fallbackText }: { answer: Answer | null; fallbackText: string }) {
  const text = answer?.answer || fallbackText;
  if (!answer) {
    return (
      <div className="rounded-xl border border-line bg-raised px-4 py-3 text-[14px] leading-relaxed shadow-panel">
        <p className="whitespace-pre-wrap">
          <TimeText text={text || "No answer was returned."} />
        </p>
      </div>
    );
  }
  const conf = Math.min(1, Math.max(0, answer.confidence ?? 0));

  return (
    <article className="overflow-hidden rounded-xl border border-line bg-raised shadow-panel animate-rise">
      <div className="px-4 pb-3 pt-3.5">
        {(answer.verdict !== "n/a" || Number.isFinite(answer.confidence)) && (
          <div className="mb-2.5 flex items-center gap-3">
            <VerdictChip verdict={answer.verdict} />
            <div className="flex min-w-0 flex-1 items-center gap-2" title="Agent's confidence in this answer">
              <span className="text-[12px] text-muted">Confidence</span>
              <div
                className="h-1.5 min-w-[48px] flex-1 overflow-hidden rounded-full bg-sunken"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(conf * 100)}
                aria-label="Confidence"
              >
                <div className={cn("h-full rounded-full", confidenceTone(conf))} style={{ width: `${conf * 100}%` }} />
              </div>
              <span className="text-[12px] font-semibold text-ink tnum">{formatPercent(conf)}</span>
            </div>
          </div>
        )}

        <p className="whitespace-pre-wrap text-[14.5px] leading-[1.6] text-ink">
          <TimeText text={text} />
        </p>

        {answer.coverage && (
          <p className="mt-2.5 flex items-start gap-1.5 text-[12px] leading-snug text-muted">
            <ScanSearch className="mt-px size-3.5 shrink-0" aria-hidden />
            <span>{answer.coverage}</span>
          </p>
        )}
      </div>

      {answer.evidence.length > 0 && (
        <div className="border-t border-line bg-surface px-3 pb-3 pt-2.5">
          <p className="mb-2 px-1 text-[12px] font-medium text-ink-2">
            Evidence <span className="font-normal text-muted tnum">({answer.evidence.length})</span>
          </p>
          <ul className="grid grid-cols-2 gap-2">
            {answer.evidence.map((ev, i) => (
              <EvidenceCard key={i} ev={ev} />
            ))}
          </ul>
        </div>
      )}

      {answer.tools_used.length > 0 && (
        <footer className="flex items-start gap-1.5 border-t border-line px-4 py-2 text-[11.5px] text-muted">
          <Wrench className="mt-px size-3 shrink-0" aria-hidden />
          <span>
            <span className="text-ink-2">Tools used:</span> {answer.tools_used.join(", ")}
          </span>
        </footer>
      )}
    </article>
  );
}

function EvidenceCard({ ev }: { ev: Evidence }) {
  const { seek, highlight } = usePlayer();
  const [failed, setFailed] = useState(false);
  const active = highlight?.t_start === ev.t_start && highlight?.t_end === ev.t_end;
  return (
    <li>
      <button
        type="button"
        onClick={() => seek(ev.t_start, { range: { t_start: ev.t_start, t_end: ev.t_end } })}
        className={cn(
          "group flex w-full flex-col overflow-hidden rounded-lg border bg-raised text-left transition-[border-color,box-shadow]",
          active ? "border-mark ring-2 ring-mark/40" : "border-line hover:border-line-strong hover:shadow-panel",
        )}
      >
        <div className="relative aspect-video w-full bg-film">
          {ev.keyframe_url && !failed ? (
            <img
              src={ev.keyframe_url}
              alt=""
              loading="lazy"
              onError={() => setFailed(true)}
              className="size-full object-cover"
            />
          ) : (
            <div className="flex size-full items-center justify-center text-white/35">
              <ImageOff className="size-4" aria-hidden />
            </div>
          )}
          <span className="absolute bottom-1 left-1 rounded-[4px] bg-mark px-1.5 text-[11px] font-semibold leading-[18px] text-black tnum">
            {formatRange(ev.t_start, ev.t_end)}
          </span>
        </div>
        <p className="line-clamp-3 px-2 py-1.5 text-[12px] leading-snug text-ink-2 group-hover:text-ink">
          {ev.reason}
        </p>
      </button>
    </li>
  );
}

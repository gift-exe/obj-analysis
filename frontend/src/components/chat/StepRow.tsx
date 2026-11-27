import { Check, ChevronRight, Layers, X } from "lucide-react";
import { useState } from "react";
import type { Step } from "../../api/types";
import { cn } from "../../lib/format";
import { usePlayer } from "../../state/player";
import { TierBadge } from "../ui/badges";
import { Spinner } from "../ui/primitives";

function formatArg(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

export function StepRow({ step }: { step: Step }) {
  const [open, setOpen] = useState(false);
  const { showOverlay, overlayOn, overlayUrl } = usePlayer();
  const args = Object.entries(step.args ?? {});
  const viewing = overlayOn && overlayUrl === step.overlay_url;

  return (
    <li className="animate-rise">
      <div className="flex items-start gap-2">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="group flex min-w-0 flex-1 items-center gap-2 rounded-md py-1 pr-1 text-left hover:bg-sunken/60"
        >
          <ChevronRight
            className={cn("size-3.5 shrink-0 text-muted transition-transform duration-150", open && "rotate-90")}
            aria-hidden
          />
          <TierBadge tier={step.tier} />
          <span
            className={cn(
              "min-w-0 flex-1 truncate text-[13px]",
              step.status === "running" ? "text-ink" : "text-ink-2",
            )}
            title={step.label}
          >
            {step.label}
          </span>
          <StatusIcon status={step.status} />
        </button>
      </div>

      {step.overlay_url && (
        <div className="ml-[22px] mt-0.5">
          <button
            type="button"
            onClick={() => step.overlay_url && showOverlay(step.overlay_url)}
            className={cn(
              "inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-[12px] font-medium transition-colors",
              viewing ? "bg-tier-gpu text-white" : "bg-tier-gpu-soft text-tier-gpu hover:brightness-95",
            )}
          >
            <Layers className="size-3.5" aria-hidden />
            {viewing ? "Showing overlay" : "View overlay"}
          </button>
        </div>
      )}

      {open && (
        <div className="ml-[22px] mt-1 mb-1.5 space-y-2 rounded-lg border border-line bg-surface px-3 py-2.5 text-[12px] animate-rise">
          <div>
            <p className="text-muted">Result</p>
            <p className="mt-0.5 text-ink">
              {step.summary ?? (step.status === "running" ? "Waiting for the tool to finish…" : "No summary returned.")}
            </p>
          </div>
          <div>
            <p className="text-muted">
              Tool <span className="text-ink-2">{step.tool}</span>
            </p>
            {args.length > 0 ? (
              <dl className="mt-1 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                {args.map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-muted">{k}</dt>
                    <dd className="break-words text-ink-2">{formatArg(v)}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="mt-0.5 text-muted">No arguments</p>
            )}
          </div>
        </div>
      )}
    </li>
  );
}

function StatusIcon({ status }: { status: Step["status"] }) {
  if (status === "running") return <Spinner className="size-3.5 shrink-0 text-accent" />;
  if (status === "failed")
    return <X className="size-3.5 shrink-0 text-bad" strokeWidth={3} aria-label="Failed" />;
  return <Check className="size-3.5 shrink-0 text-ok" strokeWidth={3} aria-label="Done" />;
}

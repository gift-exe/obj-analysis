import { AlertCircle, Check, CircleHelp, X } from "lucide-react";
import type { Tier, Verdict, Video } from "../../api/types";
import { TIER_LABEL, TIER_STYLE } from "../../lib/tiers";
import { cn, formatPercent } from "../../lib/format";
import { Spinner } from "./primitives";

/** Cost tier of a tool call. Width is fixed so step labels align. */
export function TierBadge({ tier, className }: { tier: Tier; className?: string }) {
  const known = tier in TIER_STYLE;
  return (
    <span
      className={cn(
        "inline-flex h-5 w-[52px] shrink-0 items-center justify-center rounded-[5px] text-[11px] font-semibold",
        known ? TIER_STYLE[tier] : "bg-sunken text-muted",
        className,
      )}
    >
      {known ? TIER_LABEL[tier] : String(tier)}
    </span>
  );
}

export function VerdictChip({ verdict }: { verdict: Verdict }) {
  if (verdict === "n/a") return null;
  const cfg = {
    yes: { label: "Yes", icon: Check, cls: "bg-ok-soft text-ok border-ok/30" },
    no: { label: "No", icon: X, cls: "bg-bad-soft text-bad border-bad/30" },
    uncertain: { label: "Uncertain", icon: CircleHelp, cls: "bg-warn-soft text-warn border-warn/30" },
  }[verdict];
  if (!cfg) return null;
  const Icon = cfg.icon;
  return (
    <span
      className={cn(
        "inline-flex h-6 items-center gap-1 rounded-full border px-2.5 text-[12px] font-semibold",
        cfg.cls,
      )}
    >
      <Icon className="size-3.5" strokeWidth={2.5} aria-hidden />
      {cfg.label}
    </span>
  );
}

/** Video lifecycle badge used in the library list. */
export function StatusBadge({ video, uploadProgress }: { video: Video; uploadProgress?: number }) {
  const base = "inline-flex h-5 shrink-0 items-center gap-1 whitespace-nowrap rounded-[5px] px-1.5 text-[11px] font-semibold tnum";
  switch (video.status) {
    case "uploading":
      return (
        <span className={cn(base, "bg-accent-soft text-accent")}>
          <Spinner className="size-3" />
          Uploading{uploadProgress != null ? ` ${formatPercent(uploadProgress)}` : ""}
        </span>
      );
    case "queued":
      return <span className={cn(base, "bg-sunken text-ink-2")}>Queued</span>;
    case "processing":
      return (
        <span className={cn(base, "bg-mark-soft text-mark-ink")}>
          <Spinner className="size-3" />
          Processing {formatPercent(video.progress)}
        </span>
      );
    case "ready":
      return (
        <span className={cn(base, "bg-ok-soft text-ok")}>
          <Check className="size-3" strokeWidth={3} aria-hidden />
          Ready
        </span>
      );
    case "failed":
      return (
        <span className={cn(base, "bg-bad-soft text-bad")}>
          <AlertCircle className="size-3" strokeWidth={2.5} aria-hidden />
          Failed
        </span>
      );
    default:
      return <span className={cn(base, "bg-sunken text-muted")}>{String(video.status)}</span>;
  }
}

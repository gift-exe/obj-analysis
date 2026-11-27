import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Box, Car, ChevronDown, Circle, MapPin, RotateCw, Sparkles, User } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { errorMessage, keys } from "../../api/queries";
import type { AnalysisReport as Report, EntityKind } from "../../api/types";
import { cn, formatRange, formatRelative, formatTime } from "../../lib/format";
import { usePlayer } from "../../state/player";
import { Button, ErrorNote, IconButton, Skeleton } from "../ui/primitives";

const KIND_ICON: Record<EntityKind, typeof User> = {
  person: User,
  object: Box,
  vehicle: Car,
  place: MapPin,
  other: Circle,
};

export function AnalysisSection({
  videoId,
  onAsk,
  askDisabled,
}: {
  videoId: string;
  onAsk: (q: string) => void;
  askDisabled: boolean;
}) {
  const qc = useQueryClient();
  const { setKeyMoments } = usePlayer();
  // Read-only view of the cached report; the button below fills it.
  const cached = useQuery({
    queryKey: keys.analysis(videoId),
    queryFn: () => api.runAnalysis(videoId),
    enabled: false,
    staleTime: Infinity,
  });
  const run = useMutation({
    mutationFn: (refresh: boolean) => api.runAnalysis(videoId, refresh),
    onSuccess: (report) => qc.setQueryData(keys.analysis(videoId), report),
  });
  const report = cached.data;
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    setKeyMoments(report?.key_moments ?? []);
  }, [report, setKeyMoments]);

  if (run.isPending && !report) {
    return (
      <div className="rounded-xl border border-line bg-raised p-4 shadow-panel" aria-busy="true">
        <p className="flex items-center gap-2 text-[13px] font-medium text-ink">
          <Sparkles className="size-4 text-accent animate-pulse-soft" aria-hidden />
          Watching the whole video…
        </p>
        <div className="mt-3 space-y-2">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-[92%]" />
          <Skeleton className="h-3 w-[70%]" />
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="rounded-xl border border-dashed border-line-strong px-4 py-3.5">
        <div className="flex items-center gap-3">
          <p className="min-w-0 flex-1 text-[12.5px] leading-snug text-muted">
            Get a summary of the whole video: what happens, key moments, and who or what appears.
          </p>
          <Button
            size="sm"
            variant="secondary"
            icon={<Sparkles className="size-3.5 text-accent" aria-hidden />}
            onClick={() => run.mutate(false)}
          >
            General analysis
          </Button>
        </div>
        {run.isError && (
          <ErrorNote
            className="mt-3"
            title="Analysis failed"
            message={errorMessage(run.error)}
            onRetry={() => run.mutate(false)}
          />
        )}
      </div>
    );
  }

  return (
    <ReportCard
      report={report}
      collapsed={collapsed}
      onToggle={() => setCollapsed((c) => !c)}
      refreshing={run.isPending}
      onRefresh={() => run.mutate(true)}
      refreshError={run.isError ? errorMessage(run.error) : null}
      onAsk={onAsk}
      askDisabled={askDisabled}
    />
  );
}

function ReportCard({
  report,
  collapsed,
  onToggle,
  refreshing,
  onRefresh,
  refreshError,
  onAsk,
  askDisabled,
}: {
  report: Report;
  collapsed: boolean;
  onToggle: () => void;
  refreshing: boolean;
  onRefresh: () => void;
  refreshError: string | null;
  onAsk: (q: string) => void;
  askDisabled: boolean;
}) {
  const { seek, highlight } = usePlayer();

  return (
    <article className="overflow-hidden rounded-xl border border-line bg-raised shadow-panel animate-rise">
      <header className="flex items-center gap-2 py-2 pl-4 pr-2">
        <Sparkles className="size-4 shrink-0 text-accent" aria-hidden />
        <h2 className="font-display text-[15px] font-semibold tracking-[-0.01em]">Video summary</h2>
        <span className="truncate text-[11.5px] text-muted">{formatRelative(report.generated_at)}</span>
        <div className="ml-auto flex items-center">
          <IconButton label="Run analysis again" onClick={onRefresh} disabled={refreshing}>
            <RotateCw className={cn("size-3.5", refreshing && "animate-spin")} />
          </IconButton>
          <IconButton label={collapsed ? "Expand summary" : "Collapse summary"} onClick={onToggle} aria-expanded={!collapsed}>
            <ChevronDown className={cn("size-4 transition-transform", collapsed && "-rotate-90")} />
          </IconButton>
        </div>
      </header>

      {!collapsed && (
        <div className="space-y-4 px-4 pb-4">
          {refreshError && <ErrorNote title="Couldn't refresh the analysis" message={refreshError} />}
          {report.narration && <p className="text-[13.5px] leading-[1.6] text-ink-2">{report.narration}</p>}

          {report.key_moments.length > 0 && (
            <section>
              <h3 className="mb-1.5 text-[12px] font-semibold text-ink-2">Key moments</h3>
              <ol className="space-y-0.5">
                {report.key_moments.map((k, i) => {
                  const active = highlight?.t_start === k.t_start && highlight?.t_end === k.t_end;
                  return (
                    <li key={i}>
                      <button
                        type="button"
                        onClick={() => seek(k.t_start, { range: { t_start: k.t_start, t_end: k.t_end } })}
                        className={cn(
                          "flex w-full gap-3 rounded-lg px-2 py-1.5 text-left transition-colors",
                          active ? "bg-accent-soft" : "hover:bg-sunken/70",
                        )}
                      >
                        <span className="mt-px w-[74px] shrink-0 text-[12px] font-semibold text-accent tnum">
                          {formatRange(k.t_start, k.t_end)}
                        </span>
                        <span className="min-w-0">
                          <span className="block text-[13px] font-medium leading-snug text-ink">{k.title}</span>
                          {k.description && (
                            <span className="mt-0.5 block text-[12px] leading-snug text-muted">{k.description}</span>
                          )}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </section>
          )}

          {report.entities.length > 0 && (
            <section>
              <h3 className="mb-1.5 text-[12px] font-semibold text-ink-2">Seen in the video</h3>
              <ul className="divide-y divide-line rounded-lg border border-line">
                {report.entities.map((e, i) => {
                  const Icon = KIND_ICON[e.kind] ?? Circle;
                  return (
                    <li key={i} className="flex items-center gap-2 px-2.5 py-1.5 text-[12.5px]">
                      <Icon className="size-3.5 shrink-0 text-muted" aria-label={e.kind} />
                      <span className="min-w-0 flex-1 truncate text-ink" title={e.name}>
                        {e.name}
                      </span>
                      <span className="flex shrink-0 items-center gap-1 text-[11.5px] text-muted tnum">
                        <button
                          type="button"
                          className="rounded px-1 hover:bg-sunken hover:text-ink"
                          onClick={() => seek(e.first_seen)}
                          title="First seen"
                        >
                          {formatTime(e.first_seen)}
                        </button>
                        <span aria-hidden>to</span>
                        <button
                          type="button"
                          className="rounded px-1 hover:bg-sunken hover:text-ink"
                          onClick={() => seek(e.last_seen)}
                          title="Last seen"
                        >
                          {formatTime(e.last_seen)}
                        </button>
                      </span>
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

          {report.suggested_questions.length > 0 && (
            <section>
              <h3 className="mb-1.5 text-[12px] font-semibold text-ink-2">Try asking</h3>
              <div className="flex flex-wrap gap-1.5">
                {report.suggested_questions.map((q, i) => (
                  <button
                    key={i}
                    type="button"
                    disabled={askDisabled}
                    onClick={() => onAsk(q)}
                    className="rounded-full border border-line bg-surface px-3 py-1 text-left text-[12.5px] leading-snug text-ink-2 transition-colors hover:border-accent hover:bg-accent-soft hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </article>
  );
}

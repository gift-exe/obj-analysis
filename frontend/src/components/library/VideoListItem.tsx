import { Film, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { Video } from "../../api/types";
import type { UploadItem } from "../../state/uploads";
import { cn, formatRelative, formatTime } from "../../lib/format";
import { StatusBadge } from "../ui/badges";
import { Spinner } from "../ui/primitives";

export function VideoListItem({
  video,
  selected,
  upload,
  onSelect,
  onDelete,
  deleting,
}: {
  video: Video;
  selected: boolean;
  upload?: UploadItem;
  onSelect: () => void;
  onDelete: () => void;
  deleting: boolean;
}) {
  const [confirming, setConfirming] = useState(false);
  const [failedThumb, setFailedThumb] = useState<string | null>(null);

  useEffect(() => {
    if (!confirming) return;
    const t = window.setTimeout(() => setConfirming(false), 3500);
    return () => window.clearTimeout(t);
  }, [confirming]);

  const showThumb = video.thumbnail_url && video.thumbnail_url !== failedThumb;
  const inFlight = video.status === "processing" || video.status === "queued";

  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected ? "true" : undefined}
        className={cn(
          "flex w-full items-center gap-3 rounded-lg p-2 pr-9 text-left transition-colors",
          selected ? "bg-raised shadow-panel ring-1 ring-line" : "hover:bg-sunken/70",
        )}
      >
        <div className="relative h-12 w-[76px] shrink-0 overflow-hidden rounded-md bg-film">
          {showThumb ? (
            <img
              src={video.thumbnail_url ?? undefined}
              alt=""
              loading="lazy"
              onError={() => setFailedThumb(video.thumbnail_url)}
              className="size-full object-cover"
            />
          ) : (
            <div className="flex size-full items-center justify-center text-white/40">
              <Film className="size-5" strokeWidth={1.5} aria-hidden />
            </div>
          )}
          {video.duration_s != null && (
            <span className="absolute bottom-0.5 right-0.5 rounded-[3px] bg-black/70 px-1 text-[10.5px] font-medium leading-4 text-white tnum">
              {formatTime(video.duration_s)}
            </span>
          )}
          {inFlight && (
            <span
              className="absolute inset-x-0 bottom-0 h-[3px] bg-mark transition-[width] duration-500"
              style={{ width: `${Math.round(video.progress * 100)}%` }}
              aria-hidden
            />
          )}
          {selected && <span className="absolute inset-y-0 left-0 w-[3px] bg-accent" aria-hidden />}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13.5px] font-medium leading-tight text-ink" title={video.title}>
            {video.title || video.filename}
          </p>
          <div className="mt-1.5 flex items-center gap-2">
            <StatusBadge video={video} uploadProgress={upload?.progress} />
            {/* The badge carries the story while work is in flight; show age otherwise. */}
            {(video.status === "ready" || video.status === "failed") && (
              <span className="truncate text-[11.5px] text-muted">{formatRelative(video.created_at)}</span>
            )}
          </div>
        </div>
      </button>

      <div className="absolute right-1.5 top-1/2 -translate-y-1/2">
        {deleting ? (
          <span className="flex size-7 items-center justify-center text-muted">
            <Spinner />
          </span>
        ) : confirming ? (
          <button
            type="button"
            onClick={() => {
              setConfirming(false);
              onDelete();
            }}
            className="h-7 rounded-md bg-bad px-2 text-[11.5px] font-semibold text-white shadow-panel animate-rise"
          >
            Delete
          </button>
        ) : (
          <button
            type="button"
            aria-label={`Delete ${video.title}`}
            title="Delete video"
            onClick={() => setConfirming(true)}
            className="flex size-7 items-center justify-center rounded-md text-muted opacity-0 transition-opacity hover:bg-bad-soft hover:text-bad focus-visible:opacity-100 group-hover:opacity-100"
          >
            <Trash2 className="size-3.5" />
          </button>
        )}
      </div>
    </li>
  );
}

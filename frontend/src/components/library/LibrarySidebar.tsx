import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Film, Moon, Sun, Upload, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { api } from "../../api/client";
import { errorMessage, keys, useVideos } from "../../api/queries";
import type { Video } from "../../api/types";
import { cn, formatBytes, formatPercent } from "../../lib/format";
import { useSelection } from "../../state/selection";
import { useTheme } from "../../state/theme";
import { useUploads, type UploadItem } from "../../state/uploads";
import { Button, EmptyState, ErrorNote, IconButton, ProgressBar, Skeleton } from "../ui/primitives";
import { VideoListItem } from "./VideoListItem";

export function LibrarySidebar() {
  const { theme, toggle } = useTheme();
  const { selectedId, select } = useSelection();
  const videos = useVideos();
  const { uploads, startUploads, dismiss } = useUploads();
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteVideo(id),
    onSuccess: (_d, id) => {
      qc.setQueryData<Video[]>(keys.videos, (prev) => prev?.filter((v) => v.id !== id));
      if (selectedId === id) select(null);
    },
  });

  const onFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    startUploads(Array.from(list), (id) => select(id));
  };

  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer.types).includes("Files");

  const dragHandlers = {
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    },
    onDragLeave: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDrop: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      onFiles(e.dataTransfer.files);
    },
  };

  const list = videos.data ?? [];

  return (
    <aside
      {...dragHandlers}
      aria-label="Video library"
      className="relative flex min-h-0 flex-col border-r border-line bg-surface"
    >
      <header className="flex h-14 shrink-0 items-center justify-between gap-2 px-4">
        <div className="flex items-center gap-2">
          <Logo />
          <span className="font-display text-[16px] font-semibold tracking-[-0.02em]">obj-analysis</span>
        </div>
        <IconButton label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} onClick={toggle}>
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </IconButton>
      </header>

      <div className="shrink-0 space-y-2 px-3 pb-3">
        <Button
          variant="primary"
          className="w-full"
          icon={<Upload className="size-4" aria-hidden />}
          onClick={() => fileInput.current?.click()}
        >
          Upload video
        </Button>
        <input
          ref={fileInput}
          type="file"
          accept="video/*,.mkv,.mov,.mp4,.webm"
          multiple
          className="hidden"
          onChange={(e) => {
            onFiles(e.target.files);
            e.target.value = "";
          }}
        />
        {uploads.length > 0 && (
          <ul className="space-y-1.5" aria-label="Uploads">
            {uploads.map((u) => (
              <UploadRow key={u.key} item={u} onDismiss={() => dismiss(u.key)} />
            ))}
          </ul>
        )}
      </div>

      <div className="flex shrink-0 items-baseline justify-between border-t border-line px-4 pb-1.5 pt-3">
        <h2 className="text-[12.5px] font-semibold text-ink-2">Library</h2>
        {list.length > 0 && <span className="text-[12px] text-muted tnum">{list.length}</span>}
      </div>

      <div className="scroll-quiet min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {videos.isPending ? (
          <div className="space-y-1 px-1 pt-1" aria-busy="true">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="flex gap-3 rounded-lg p-2">
                <Skeleton className="h-12 w-[76px] shrink-0" />
                <div className="flex-1 space-y-2 py-1">
                  <Skeleton className="h-3 w-4/5" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
            ))}
          </div>
        ) : videos.isError ? (
          <ErrorNote
            className="mx-1 mt-1"
            title="Couldn't load the library"
            message={errorMessage(videos.error)}
            onRetry={() => void videos.refetch()}
          />
        ) : list.length === 0 ? (
          <EmptyState
            icon={<Film className="size-7" strokeWidth={1.5} />}
            title="No videos yet"
            body="Upload a video, or drop one anywhere on this panel, to build its searchable memory."
          />
        ) : (
          <ul className="space-y-0.5">
            {list.map((v) => (
              <VideoListItem
                key={v.id}
                video={v}
                selected={v.id === selectedId}
                upload={uploads.find((u) => u.videoId === v.id && u.phase === "uploading")}
                onSelect={() => select(v.id)}
                onDelete={() => remove.mutate(v.id)}
                deleting={remove.isPending && remove.variables === v.id}
              />
            ))}
          </ul>
        )}
        {remove.isError && (
          <ErrorNote className="mx-1 mt-2" title="Couldn't delete the video" message={errorMessage(remove.error)} />
        )}
      </div>

      {dragging && (
        <div className="pointer-events-none absolute inset-2 z-20 flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-accent bg-accent-soft/90 text-accent backdrop-blur-[2px] animate-rise">
          <Upload className="mb-2 size-6" aria-hidden />
          <p className="font-display text-[16px] font-semibold">Drop to upload</p>
          <p className="text-[12.5px] opacity-80">MP4, MOV, MKV or WebM</p>
        </div>
      )}
    </aside>
  );
}

function UploadRow({ item, onDismiss }: { item: UploadItem; onDismiss: () => void }) {
  const phaseLabel: Record<UploadItem["phase"], string> = {
    creating: "Preparing…",
    uploading: formatPercent(item.progress),
    starting: "Starting ingest…",
    done: "Uploaded",
    error: "Failed",
  };
  return (
    <li className="rounded-lg border border-line bg-raised px-2.5 py-2 animate-rise">
      <div className="flex items-center gap-2">
        <p className="min-w-0 flex-1 truncate text-[12.5px] font-medium" title={item.filename}>
          {item.filename}
        </p>
        <span className={cn("shrink-0 text-[12px] tnum", item.phase === "error" ? "text-bad" : "text-muted")}>
          {phaseLabel[item.phase]}
        </span>
        {item.phase === "error" && (
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Dismiss"
            className="-mr-1 rounded p-0.5 text-muted hover:bg-sunken hover:text-ink"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>
      {item.phase === "error" ? (
        <p className="mt-1 text-[12px] leading-snug text-bad">{item.error}</p>
      ) : (
        <>
          <ProgressBar
            className="mt-1.5"
            value={item.phase === "creating" ? 1 : item.progress}
            tone={item.phase === "done" ? "ok" : item.phase === "creating" ? "muted" : "accent"}
            striped={item.phase === "starting" || item.phase === "creating"}
            label={`Uploading ${item.filename}`}
          />
          <p className="mt-1 text-[11.5px] text-muted tnum">{formatBytes(item.size)}</p>
        </>
      )}
    </li>
  );
}

function Logo() {
  return (
    <svg viewBox="0 0 24 24" className="size-6" aria-hidden>
      <rect width="24" height="24" rx="6" className="fill-ink" />
      <rect x="5" y="6.5" width="14" height="11" rx="1.5" fill="none" stroke="var(--c-mark)" strokeWidth="2" />
      <rect x="9" y="9.5" width="4" height="5" className="fill-accent" />
    </svg>
  );
}

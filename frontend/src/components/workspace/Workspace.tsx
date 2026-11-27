import { Clapperboard, Mic, MicOff } from "lucide-react";
import type { ReactNode } from "react";
import type { Video } from "../../api/types";
import { formatResolution, formatTime } from "../../lib/format";
import { StatusBadge } from "../ui/badges";
import { EmptyState } from "../ui/primitives";
import { Player } from "./Player";
import { ProcessingView } from "./ProcessingView";
import { TimelineStrip } from "./Timeline";

export function Workspace({ video }: { video: Video }) {
  const ready = video.status === "ready";
  const resolution = formatResolution(video.width, video.height);

  return (
    <main className="flex min-h-0 min-w-0 flex-col" aria-label="Video workspace">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line px-6">
        <h1
          className="min-w-0 truncate font-display text-[19px] font-semibold tracking-[-0.02em]"
          title={video.title}
        >
          {video.title || video.filename}
        </h1>
        {!ready && <StatusBadge video={video} />}
        <dl className="ml-auto flex shrink-0 items-center gap-4 text-[12.5px] text-muted tnum">
          {video.duration_s != null && <Meta label="Duration">{formatTime(video.duration_s)}</Meta>}
          {resolution && <Meta label="Resolution">{resolution}</Meta>}
          {video.fps != null && <Meta label="Frame rate">{Math.round(video.fps * 100) / 100} fps</Meta>}
          {video.has_audio != null && (
            <Meta label="Audio">
              {video.has_audio ? (
                <Mic className="size-3.5" aria-label="Has audio" />
              ) : (
                <MicOff className="size-3.5" aria-label="No audio" />
              )}
            </Meta>
          )}
        </dl>
      </header>

      <div className="scroll-quiet min-h-0 flex-1 overflow-y-auto">
        {ready ? (
          video.playback_url ? (
            <div
              className="mx-auto w-full px-6 pb-8 pt-5"
              // Keep player + timeline within the viewport height on laptops.
              style={{ maxWidth: "max(560px, calc((100vh - 260px) * 16 / 9 + 48px))" }}
            >
              <Player video={video} />
              <div className="mt-5">
                <TimelineStrip video={video} />
              </div>
            </div>
          ) : (
            <EmptyState
              className="h-full"
              icon={<Clapperboard className="size-8" strokeWidth={1.5} />}
              title="No playback file"
              body="The video is marked ready, but the server didn't return a playback URL."
            />
          )
        ) : (
          <ProcessingView video={video} />
        )}
      </div>
    </main>
  );
}

function Meta({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center" title={label}>
      <dt className="sr-only">{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

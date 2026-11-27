import { Film, Upload } from "lucide-react";
import { useEffect, useRef } from "react";
import { errorMessage, useVideos } from "./api/queries";
import { ChatPanel } from "./components/chat/ChatPanel";
import { LibrarySidebar } from "./components/library/LibrarySidebar";
import { Button, EmptyState, ErrorNote, Skeleton } from "./components/ui/primitives";
import { Workspace } from "./components/workspace/Workspace";
import { ObjectSelectionProvider } from "./state/objectSelection";
import { PlayerProvider } from "./state/player";
import { useSelection } from "./state/selection";
import { useUploads } from "./state/uploads";

export function App() {
  const videos = useVideos();
  const { selectedId, select } = useSelection();
  const list = videos.data;
  const selected = list?.find((v) => v.id === selectedId) ?? null;

  // Fall back to the newest video when nothing (or a missing video) is selected.
  useEffect(() => {
    if (!list) return;
    if (selectedId && list.some((v) => v.id === selectedId)) return;
    select(list[0]?.id ?? null);
  }, [list, selectedId, select]);

  return (
    <div className="grid h-full min-w-[960px] grid-cols-[236px_minmax(0,1fr)_340px] xl:grid-cols-[272px_minmax(0,1fr)_392px] 2xl:grid-cols-[288px_minmax(0,1fr)_440px]">
      <LibrarySidebar />
      {selected ? (
        <PlayerProvider key={selected.id}>
          <ObjectSelectionProvider videoId={selected.id}>
            <Workspace video={selected} />
            <ChatPanel video={selected} />
          </ObjectSelectionProvider>
        </PlayerProvider>
      ) : (
        <div className="col-span-2 flex min-h-0 items-center justify-center">
          {videos.isPending ? (
            <div className="w-full max-w-[640px] space-y-4 px-6" aria-busy="true">
              <Skeleton className="aspect-video w-full rounded-xl" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : videos.isError ? (
            <ErrorNote
              className="max-w-md"
              title="Couldn't reach the video library"
              message={errorMessage(videos.error)}
              onRetry={() => void videos.refetch()}
            />
          ) : (
            <Welcome onCreated={select} />
          )}
        </div>
      )}
    </div>
  );
}

function Welcome({ onCreated }: { onCreated: (id: string) => void }) {
  const { startUploads } = useUploads();
  const input = useRef<HTMLInputElement>(null);
  return (
    <EmptyState
      icon={
        <div className="flex size-14 items-center justify-center rounded-2xl bg-accent-soft text-accent">
          <Film className="size-7" strokeWidth={1.5} />
        </div>
      }
      title="Upload a video to start"
      body="It's processed once into captions, keyframes and a search index. Then you can ask questions and get answers with timestamped evidence."
      action={
        <>
          <Button variant="primary" icon={<Upload className="size-4" aria-hidden />} onClick={() => input.current?.click()}>
            Upload video
          </Button>
          <input
            ref={input}
            type="file"
            accept="video/*,.mkv,.mov,.mp4,.webm"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) startUploads(Array.from(e.target.files), onCreated);
              e.target.value = "";
            }}
          />
        </>
      }
    />
  );
}

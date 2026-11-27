import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import { keys, patchVideoInList, errorMessage } from "../api/queries";
import { putFile } from "../api/upload";

export type UploadPhase = "creating" | "uploading" | "starting" | "done" | "error";

export interface UploadItem {
  key: string;
  filename: string;
  size: number;
  videoId: string | null;
  progress: number;
  phase: UploadPhase;
  error: string | null;
}

const VIDEO_EXT = /\.(mp4|m4v|mov|mkv|webm|avi|mpe?g|ts|mts|wmv|flv|3gp)$/i;

export function isVideoFile(file: File): boolean {
  return file.type.startsWith("video/") || VIDEO_EXT.test(file.name);
}

interface UploadsContextValue {
  uploads: UploadItem[];
  /** Start uploading files. `onCreated` fires with the first new video id. */
  startUploads: (files: File[], onCreated?: (videoId: string) => void) => void;
  dismiss: (key: string) => void;
  byVideoId: (videoId: string) => UploadItem | undefined;
}

const UploadsContext = createContext<UploadsContextValue | null>(null);

let seq = 0;

export function UploadsProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [uploads, setUploads] = useState<UploadItem[]>([]);

  const patch = useCallback((key: string, p: Partial<UploadItem>) => {
    setUploads((list) => list.map((u) => (u.key === key ? { ...u, ...p } : u)));
  }, []);

  const runOne = useCallback(
    async (file: File, key: string, onCreated?: (id: string) => void) => {
      try {
        const created = await api.createVideo({
          filename: file.name,
          content_type: file.type || "application/octet-stream",
          size: file.size,
        });
        const id = created.video.id;
        patch(key, { videoId: id, phase: "uploading" });
        patchVideoInList(qc, created.video);
        onCreated?.(id);

        let last = 0;
        await putFile({
          url: created.upload_url,
          method: created.upload_method,
          headers: created.upload_headers ?? {},
          file,
          onProgress: (f) => {
            // Throttle state updates to whole percents.
            if (f === 1 || f - last >= 0.01) {
              last = f;
              patch(key, { progress: f });
            }
          },
        });

        patch(key, { phase: "starting", progress: 1 });
        const video = await api.ingestVideo(id);
        patchVideoInList(qc, video);
        patch(key, { phase: "done" });
        void qc.invalidateQueries({ queryKey: keys.videos });
        window.setTimeout(() => setUploads((l) => l.filter((u) => u.key !== key)), 1600);
      } catch (err) {
        patch(key, { phase: "error", error: errorMessage(err) });
        void qc.invalidateQueries({ queryKey: keys.videos });
      }
    },
    [patch, qc],
  );

  const startUploads = useCallback<UploadsContextValue["startUploads"]>(
    (files, onCreated) => {
      let notified = false;
      const items: UploadItem[] = files.map((file) => {
        const ok = isVideoFile(file);
        return {
          key: `u${++seq}`,
          filename: file.name,
          size: file.size,
          videoId: null,
          progress: 0,
          phase: ok ? "creating" : "error",
          error: ok ? null : "This file isn't a video. Choose an MP4, MOV, MKV or WebM file.",
        };
      });
      setUploads((list) => [...items, ...list]);
      files.forEach((file, i) => {
        const item = items[i];
        if (!item || item.phase === "error") return;
        void runOne(file, item.key, (id) => {
          if (!notified) {
            notified = true;
            onCreated?.(id);
          }
        });
      });
    },
    [runOne],
  );

  const dismiss = useCallback((key: string) => setUploads((l) => l.filter((u) => u.key !== key)), []);

  const value = useMemo<UploadsContextValue>(
    () => ({
      uploads,
      startUploads,
      dismiss,
      byVideoId: (id) => uploads.find((u) => u.videoId === id && u.phase !== "done"),
    }),
    [uploads, startUploads, dismiss],
  );

  return <UploadsContext.Provider value={value}>{children}</UploadsContext.Provider>;
}

export function useUploads() {
  const ctx = useContext(UploadsContext);
  if (!ctx) throw new Error("useUploads must be used inside <UploadsProvider>");
  return ctx;
}

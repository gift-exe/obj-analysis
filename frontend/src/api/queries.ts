import { useQuery, type QueryClient } from "@tanstack/react-query";
import { api, ApiError } from "./client";
import type { Video, VideoStatus } from "./types";

export const keys = {
  videos: ["video-list"] as const,
  steps: (id: string) => ["steps", id] as const,
  timeline: (id: string) => ["timeline", id] as const,
  analysis: (id: string) => ["analysis", id] as const,
  toolRuns: (id: string) => ["tool-runs", id] as const,
  conversations: (videoId: string) => ["conversations", videoId] as const,
  messages: (conversationId: string) => ["messages", conversationId] as const,
};

export const isTerminal = (s: VideoStatus) => s === "ready" || s === "failed";

const POLL_MS = 3000;

export function useVideos() {
  return useQuery({
    queryKey: keys.videos,
    queryFn: api.listVideos,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.some((v) => !isTerminal(v.status)) ? POLL_MS : false;
    },
  });
}

/** Replace one video inside the cached list (used by live SSE updates). */
export function patchVideoInList(qc: QueryClient, video: Video) {
  qc.setQueryData<Video[]>(keys.videos, (prev) => {
    if (!prev) return prev;
    const idx = prev.findIndex((v) => v.id === video.id);
    if (idx === -1) return [video, ...prev];
    const next = prev.slice();
    next[idx] = video;
    return next;
  });
}

export function useSteps(videoId: string, poll: boolean) {
  return useQuery({
    queryKey: keys.steps(videoId),
    queryFn: () => api.getSteps(videoId),
    refetchInterval: poll ? POLL_MS : false,
  });
}

export function useTimeline(videoId: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.timeline(videoId),
    queryFn: () => api.getTimeline(videoId),
    enabled,
    staleTime: Infinity,
    // 409 means "not ready yet"; retry a few times with backoff.
    retry: (count, err) => (err instanceof ApiError && err.status === 409 ? count < 5 : count < 2),
  });
}

export function useToolRuns(videoId: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.toolRuns(videoId),
    queryFn: () => api.listToolRuns(videoId),
    enabled,
  });
}

export function useConversations(videoId: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.conversations(videoId),
    queryFn: () => api.listConversations(videoId),
    enabled,
  });
}

export function useMessages(conversationId: string | null) {
  return useQuery({
    queryKey: keys.messages(conversationId ?? "none"),
    queryFn: () => api.listMessages(conversationId as string),
    enabled: conversationId !== null,
  });
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamMessage } from "../api/client";
import { errorMessage, keys, useConversations, useMessages } from "../api/queries";
import type { Conversation, Message, ObjectSelection, Step } from "../api/types";

/** The in-flight turn, shown until the `final` event lands in the message cache. */
export interface PendingTurn {
  userText: string;
  selection: ObjectSelection | null;
  /** True once the server echoed the stored user message (it's then in the cache). */
  userCommitted: boolean;
  plan: string | null;
  steps: Step[];
  answerText: string;
  error: string | null;
}

export function useChat(videoId: string, enabled: boolean) {
  const qc = useQueryClient();
  const conversations = useConversations(videoId, enabled);
  // Reuse the newest conversation for this video (the list is newest first).
  const conversationId = conversations.data?.[0]?.id ?? null;
  const messages = useMessages(conversationId);

  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const appendMessage = useCallback(
    (cid: string, msg: Message) => {
      qc.setQueryData<Message[]>(keys.messages(cid), (prev = []) =>
        prev.some((m) => m.id === msg.id) ? prev.map((m) => (m.id === msg.id ? msg : m)) : [...prev, msg],
      );
    },
    [qc],
  );

  const send = useCallback(
    async (raw: string, selection: ObjectSelection | null = null) => {
      const content = raw.trim();
      if (!content || streaming) return;

      const update = (fn: (p: PendingTurn) => PendingTurn) => setPending((p) => (p ? fn(p) : p));
      setPending({
        userText: content,
        selection,
        userCommitted: false,
        plan: null,
        steps: [],
        answerText: "",
        error: null,
      });
      setStreaming(true);
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      let cid = conversationId;
      let gotFinal = false;
      let gotError = false;
      try {
        if (!cid) {
          const conv = await api.createConversation(videoId);
          qc.setQueryData<Message[]>(keys.messages(conv.id), []);
          qc.setQueryData<Conversation[]>(keys.conversations(videoId), (prev = []) => [conv, ...prev]);
          cid = conv.id;
        }
        const target = cid;
        await streamMessage(
          target,
          content,
          (ev) => {
            switch (ev.type) {
              case "user":
                appendMessage(target, ev.data.message);
                update((p) => ({ ...p, userCommitted: true }));
                break;
              case "plan":
                update((p) => ({ ...p, plan: ev.data.text }));
                break;
              case "tool_call":
              case "tool_result": {
                const step = ev.data.step;
                update((p) => {
                  const idx = p.steps.findIndex((s) => s.id === step.id);
                  const steps = idx === -1 ? [...p.steps, step] : p.steps.map((s, i) => (i === idx ? step : s));
                  return { ...p, steps };
                });
                break;
              }
              case "answer_delta":
                update((p) => ({ ...p, answerText: p.answerText + ev.data.text }));
                break;
              case "final":
                gotFinal = true;
                appendMessage(target, ev.data.message);
                setPending(null);
                break;
              case "error":
                gotError = true;
                update((p) => ({ ...p, error: ev.data.message || "The agent stopped with an error." }));
                break;
            }
          },
          ctrl.signal,
          selection?.id,
        );
        if (!gotFinal && !gotError) {
          update((p) => ({ ...p, error: "The connection closed before the answer was finished. Try asking again." }));
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        update((p) => ({ ...p, error: errorMessage(err) }));
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
        setStreaming(false);
        // Tool runs (and possibly new overlays) may have been created.
        void qc.invalidateQueries({ queryKey: keys.toolRuns(videoId) });
        if (!gotFinal && cid) void qc.invalidateQueries({ queryKey: keys.messages(cid) });
      }
    },
    [appendMessage, conversationId, qc, streaming, videoId],
  );

  return { conversations, conversationId, messages, pending, streaming, send };
}

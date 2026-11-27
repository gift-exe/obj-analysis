import { ChevronRight, Hourglass, MessageSquareText } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { errorMessage } from "../../api/queries";
import type { Message, ObjectSelection, Video } from "../../api/types";
import { cn, formatTime } from "../../lib/format";
import { useObjectSelection } from "../../state/objectSelection";
import { usePlayer } from "../../state/player";
import { useChat, type PendingTurn } from "../../state/useChat";
import { TIER_DOT } from "../../lib/tiers";
import { EmptyState, ErrorNote, Skeleton, Spinner } from "../ui/primitives";
import { AnalysisSection } from "./AnalysisReport";
import { AnswerCard } from "./AnswerCard";
import { Composer } from "./Composer";
import { StepImages } from "./StepImages";
import { StepRow } from "./StepRow";

export function ChatPanel({ video }: { video: Video }) {
  const ready = video.status === "ready";
  const chat = useChat(video.id, ready);
  const { setEvidence, seek } = usePlayer();
  const objSel = useObjectSelection();
  const messages = chat.messages.data;

  // Evidence markers on the timeline follow the latest answer.
  useEffect(() => {
    const latest = messages
      ?.slice()
      .reverse()
      .find((m) => m.role === "assistant" && m.answer);
    setEvidence(latest?.answer?.evidence ?? []);
  }, [messages, setEvidence]);

  // Keep the thread pinned to the bottom while the user hasn't scrolled up.
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, chat.pending]);

  const send = (text: string) => {
    stick.current = true;
    void chat.send(text, objSel.selection);
  };

  return (
    <aside aria-label="Ask about this video" className="flex min-h-0 flex-col border-l border-line bg-surface">
      <header className="flex h-14 shrink-0 items-center gap-2 px-4">
        <h2 className="font-display text-[16px] font-semibold tracking-[-0.02em]">Ask the video</h2>
        {chat.streaming && (
          <span className="ml-auto inline-flex items-center gap-1.5 text-[12px] text-accent">
            <Spinner className="size-3" />
            Agent working
          </span>
        )}
      </header>

      {!ready ? (
        <EmptyState
          className="flex-1 border-t border-line"
          icon={<Hourglass className="size-7" strokeWidth={1.5} />}
          title={video.status === "failed" ? "This video can't be searched" : "Questions open when processing ends"}
          body={
            video.status === "failed"
              ? "Processing failed, so the agent has no memory of this video to answer from."
              : "The agent answers from the video's memory, which is still being built."
          }
        />
      ) : (
        <>
          <div
            ref={scroller}
            onScroll={(e) => {
              const el = e.currentTarget;
              stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
            }}
            className="scroll-quiet min-h-0 flex-1 space-y-5 overflow-y-auto border-t border-line px-4 py-4"
          >
            <AnalysisSection videoId={video.id} onAsk={send} askDisabled={chat.streaming} />
            <Thread chat={chat} />
          </div>
          <Composer
            onSend={send}
            disabled={chat.conversations.isPending || chat.conversations.isError}
            busy={chat.streaming}
            placeholder={objSel.selection ? "Ask about the selected object…" : "Ask about this video…"}
            selection={objSel.selection}
            onClearSelection={objSel.clear}
            onShowSelection={() => objSel.selection && seek(objSel.selection.t)}
          />
        </>
      )}
    </aside>
  );
}

function Thread({ chat }: { chat: ReturnType<typeof useChat> }) {
  const { conversations, messages, pending, conversationId } = chat;

  if (conversations.isPending || (conversationId && messages.isPending)) {
    return (
      <div className="space-y-3" aria-busy="true" aria-label="Loading conversation">
        <Skeleton className="ml-auto h-9 w-3/5 rounded-xl" />
        <Skeleton className="h-24 w-full rounded-xl" />
      </div>
    );
  }
  if (conversations.isError) {
    return (
      <ErrorNote
        title="Couldn't load the conversation"
        message={errorMessage(conversations.error)}
        onRetry={() => void conversations.refetch()}
      />
    );
  }
  if (messages.isError && !messages.data) {
    return (
      <ErrorNote
        title="Couldn't load past messages"
        message={errorMessage(messages.error)}
        onRetry={() => void messages.refetch()}
      />
    );
  }

  const list = messages.data ?? [];
  if (list.length === 0 && !pending) {
    return (
      <EmptyState
        className="py-8"
        icon={<MessageSquareText className="size-6" strokeWidth={1.5} />}
        title="Ask anything about this video"
        body="For example: when does the red car first appear, or does anyone cross against the light? Answers come with timestamped evidence."
      />
    );
  }

  return (
    <ol className="space-y-5">
      {list.map((m, i) =>
        m.role === "user" ? (
          <UserBubble key={m.id} text={m.content} selection={m.selection} />
        ) : (
          <PastAnswer key={m.id} message={m} defaultOpen={i === list.length - 1 && !pending} />
        ),
      )}
      {pending && <LiveTurn turn={pending} />}
    </ol>
  );
}

function UserBubble({ text, selection }: { text: string; selection?: ObjectSelection | null }) {
  const { seek } = usePlayer();
  return (
    <li className="flex flex-col items-end gap-1 animate-rise">
      {selection && (
        <button
          type="button"
          onClick={() => seek(selection.t)}
          title="Show the selected object on the video"
          className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-raised py-1 pl-1 pr-2 text-[11.5px] text-ink-2 hover:text-ink"
        >
          {selection.crop_url && (
            <img src={selection.crop_url} alt="" className="size-6 rounded object-cover ring-1 ring-[#00e676]" />
          )}
          Object at {formatTime(selection.t)}
        </button>
      )}
      <p className="max-w-[88%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-ink px-3.5 py-2 text-[13.5px] leading-snug text-bg">
        {text}
      </p>
    </li>
  );
}

function TierDots({ steps }: { steps: Message["steps"] }) {
  return (
    <span className="flex items-center -space-x-0.5" aria-hidden>
      {steps.slice(0, 8).map((s) => (
        <span key={s.id} className={cn("size-2 rounded-full ring-2 ring-surface", TIER_DOT[s.tier] ?? "bg-muted")} />
      ))}
    </span>
  );
}

function PastAnswer({ message, defaultOpen }: { message: Message; defaultOpen: boolean }) {
  // The newest answer keeps its steps open so the live view doesn't jump on `final`.
  const [open, setOpen] = useState(defaultOpen);
  const steps = message.steps ?? [];
  return (
    <li className="space-y-2">
      {steps.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="flex items-center gap-2 rounded-md py-0.5 pr-1.5 text-[12px] text-muted hover:text-ink"
          >
            <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} aria-hidden />
            {steps.length === 1 ? "Used 1 tool" : `Used ${steps.length} tools`}
            <TierDots steps={steps} />
          </button>
          {open && (
            <ul className="mt-1 space-y-0.5">
              {steps.map((s) => (
                <StepRow key={s.id} step={s} />
              ))}
            </ul>
          )}
        </div>
      )}
      <StepImages steps={steps} />
      <AnswerCard answer={message.answer} fallbackText={message.content} />
    </li>
  );
}

function LiveTurn({ turn }: { turn: PendingTurn }) {
  const working = !turn.error;
  return (
    <>
      {!turn.userCommitted && <UserBubble text={turn.userText} selection={turn.selection} />}
      <li className="space-y-2" aria-live="polite">
        {turn.plan && (
          <p className="border-l-2 border-accent/50 pl-2.5 text-[12.5px] italic leading-snug text-ink-2 animate-rise">
            {turn.plan}
          </p>
        )}
        {turn.steps.length > 0 && (
          <ul className="space-y-0.5">
            {turn.steps.map((s) => (
              <StepRow key={s.id} step={s} />
            ))}
          </ul>
        )}
        <StepImages steps={turn.steps} />
        {turn.answerText ? (
          <div className="rounded-xl border border-line bg-raised px-4 py-3 text-[14.5px] leading-[1.6] shadow-panel">
            <span className="whitespace-pre-wrap">{turn.answerText}</span>
            {working && <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-accent animate-pulse-soft" />}
          </div>
        ) : (
          working && (
            <p className="flex items-center gap-2 text-[12.5px] text-muted">
              <Spinner className="text-accent" />
              {turn.steps.length === 0 ? "Planning…" : turn.steps.some((s) => s.status === "running") ? "Running tools…" : "Writing the answer…"}
            </p>
          )
        )}
        {turn.error && <ErrorNote title="The agent couldn't finish" message={turn.error} />}
      </li>
    </>
  );
}

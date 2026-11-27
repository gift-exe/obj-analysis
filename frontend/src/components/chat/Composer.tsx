import { ArrowUp, X } from "lucide-react";
import { useLayoutEffect, useRef, useState } from "react";
import type { ObjectSelection } from "../../api/types";
import { cn, formatTime } from "../../lib/format";
import { Spinner } from "../ui/primitives";

export function Composer({
  onSend,
  disabled,
  busy,
  placeholder,
  selection,
  onClearSelection,
  onShowSelection,
}: {
  onSend: (text: string) => void;
  disabled: boolean;
  busy: boolean;
  placeholder: string;
  /** an object clicked on the video; sent along with the next questions until cleared */
  selection?: ObjectSelection | null;
  onClearSelection?: () => void;
  onShowSelection?: () => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with content up to ~6 lines.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 148)}px`;
  }, [text]);

  const submit = () => {
    if (disabled || busy || !text.trim()) return;
    onSend(text);
    setText("");
  };

  const locked = disabled || busy;

  return (
    <form
      className="shrink-0 border-t border-line bg-surface p-3"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      {selection && (
        <div className="mb-2 flex items-center gap-2 rounded-lg border border-line bg-raised p-1.5 pr-1 animate-rise">
          <button
            type="button"
            onClick={onShowSelection}
            title="Show it on the video"
            className="flex min-w-0 flex-1 items-center gap-2 text-left"
          >
            {selection.crop_url ? (
              <img
                src={selection.crop_url}
                alt=""
                className="size-9 shrink-0 rounded-md object-cover ring-2 ring-[#00e676]"
              />
            ) : (
              <span className="size-9 shrink-0 rounded-md bg-sunken ring-2 ring-[#00e676]" />
            )}
            <span className="min-w-0 text-[12.5px] leading-tight">
              <span className="block font-medium text-ink">Selected object</span>
              <span className="block text-muted">
                at {formatTime(selection.t)} · questions are about this object
              </span>
            </span>
          </button>
          <button
            type="button"
            onClick={onClearSelection}
            aria-label="Clear the selected object"
            title="Clear the selection"
            className="flex size-7 shrink-0 items-center justify-center rounded-md text-muted hover:bg-sunken hover:text-ink"
          >
            <X className="size-4" />
          </button>
        </div>
      )}
      <div
        className={cn(
          "flex items-end gap-2 rounded-xl border bg-raised p-1.5 pl-3 transition-colors",
          locked ? "border-line" : "border-line-strong focus-within:border-accent",
        )}
      >
        <textarea
          ref={ref}
          rows={1}
          value={text}
          disabled={locked}
          placeholder={busy ? "Working on the answer…" : placeholder}
          aria-label="Ask a question about this video"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          className="max-h-[148px] min-h-[24px] flex-1 resize-none bg-transparent py-1.5 text-[14px] leading-6 text-ink outline-none placeholder:text-muted disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={locked || !text.trim()}
          aria-label="Send"
          className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-ink transition-[filter,opacity] hover:brightness-110 disabled:opacity-35"
        >
          {busy ? <Spinner className="size-4" /> : <ArrowUp className="size-4" strokeWidth={2.5} />}
        </button>
      </div>
      <p className="mt-1.5 px-1 text-[11px] text-muted">Enter to send, Shift+Enter for a new line</p>
    </form>
  );
}

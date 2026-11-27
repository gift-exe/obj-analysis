import { ExternalLink, ImageOff } from "lucide-react";
import { useState } from "react";
import type { Step } from "../../api/types";
import { TimeText } from "./TimeText";

/** Pictures and clips produced by tools (segment_scene, movement_map, make_highlight_clip...), shown inline. */
export function StepImages({ steps }: { steps: Step[] }) {
  const media = steps.filter((s) => s.image_url || s.clip_url);
  if (media.length === 0) return null;
  return (
    <div className="space-y-2">
      {media.map((s) => (
        <StepMedia key={s.id} step={s} />
      ))}
    </div>
  );
}

function StepMedia({ step }: { step: Step }) {
  const [failed, setFailed] = useState(false);
  const url = (step.clip_url ?? step.image_url)!;
  const isClip = !!step.clip_url;
  const caption = (step.summary ?? step.label).replace(/^\(cached\)\s*/, "");
  return (
    <figure className="overflow-hidden rounded-xl border border-line bg-raised shadow-panel animate-rise">
      {failed ? (
        <div className="flex items-center gap-2 px-4 py-6 text-[12.5px] text-muted">
          <ImageOff className="size-4" aria-hidden />
          {isClip ? "The clip couldn't be loaded." : "The picture couldn't be loaded."}
        </div>
      ) : isClip ? (
        <video
          src={url}
          controls
          playsInline
          preload="metadata"
          onError={() => setFailed(true)}
          className="block max-h-[70vh] w-full bg-black"
        />
      ) : (
        <a href={url} target="_blank" rel="noreferrer" title="Open full size">
          <img
            src={url}
            alt={caption}
            loading="lazy"
            onError={() => setFailed(true)}
            className="block max-h-[70vh] w-full bg-sunken object-contain"
          />
        </a>
      )}
      <figcaption className="flex items-start gap-2 px-3 py-2 text-[12px] text-muted">
        {/* clip captions list the original times: make them seek the main player */}
        <span className={isClip ? "min-w-0 flex-1 leading-relaxed" : "min-w-0 flex-1 truncate"} title={caption}>
          {isClip ? <TimeText text={caption} /> : caption}
        </span>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex shrink-0 items-center gap-1 text-ink-2 hover:text-ink"
        >
          {isClip ? "Open" : "Full size"} <ExternalLink className="size-3" aria-hidden />
        </a>
      </figcaption>
    </figure>
  );
}

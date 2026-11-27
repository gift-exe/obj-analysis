"""The intelligence layer: routes a chat message, runs the tool loop, returns a grounded answer.

Uses a JSON action protocol (one JSON object per turn) instead of native function calling, so it
works with any OpenAI-compatible server (vLLM, Ollama, hosted APIs) and any model that can emit JSON.
"""

import json
import logging
import re
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import select

from app.agent.analysis import build_report
from app.core.config import get_settings
from app.core.db import session_scope
from app.core.media import fmt_t
from app.llm import chat_json
from app.models import Conversation, Message, Video
from app.tools.memory import nearest_keyframe_url
from app.tools.registry import TOOLS, ToolOutput

log = logging.getLogger(__name__)
TIER_ORDER = {"memory": 0, "vlm": 1, "gpu": 2, "plugin": 2}

ROUTER_PROMPT = """Classify the user's message about a video.
"general": they want an overall analysis, summary, narration or description of the whole video.
"specific": anything else (a question about an event, a count, a time, a detail, a request to show or highlight
things in a picture, a follow-up).
Message: {message}
Return only JSON: {{"path": "general" | "specific"}}"""

SYSTEM_PROMPT = """You are a video analysis agent answering questions about ONE video by calling tools.
You cannot see the video yourself; you only know what the tools return.

VIDEO: "{title}", duration {duration} ({duration_s:.1f}s), {clips} indexed clips.

TOOLS (cheapest tier first; prefer the cheapest tier that can answer):
{tools}

METHOD
1. Plan: decide what evidence would answer the question and which tier is needed. Counting ("how many"),
   durations, positions, crossings and zones need GPU tools. Most "did X happen", "when" and "what" questions
   only need memory + VLM tools.
2. Locate: find candidate moments with search_moments (use 2+ phrasings for events). For short videos,
   get_timeline over the whole video is also fine.
3. Inspect: watch_clip on the most promising candidates (at most 3). Call GPU tools only when the plan needs them.
4. Verify: for a yes/no question, look for evidence AGAINST your first conclusion before answering. Before
   answering "no", make sure all scenes were covered.
5. Preconditions: before relying on a detail (a traffic light's colour, a label, a small object), first confirm it
   is actually visible from this camera (e.g. ask watch_clip/inspect_frame whether it can be seen at all). A tool
   result with "visible": false or low confidence means the detail cannot be determined from this video.
6. Consistency: if tool results contradict each other (e.g. a light is red, then green a second later, with no
   clear view), do not pick one; the answer is uncertain.
7. Counting: "how many X are in / appear in / pass through the video" means DISTINCT objects -> count_objects.
   "How many X at time t / in the first frame" -> detect_objects with t0 = t1 = that time (first frame: 0).
   max_visible_at_once is the most seen in one frame, NOT the number of distinct objects. Use concrete object
   words (car, truck, person); generic words like "vehicle" are expanded automatically. On checkout/till footage
   use checkout_scan_count for item counts ("each item type", "distinct/unique/exclusive items" all mean the
   per-item-type totals it returns).
8. Zero results: a detector finding 0 of something the video summary or captions mention means the detector
   probably missed it, not that it is absent. Say so, and check with inspect_frame or watch_clip.
9. Numbers: never add up or compute counts yourself when a tool returns a total; quote tool numbers exactly.
   When asked how a number was obtained, explain using the tool's "method" field only.
10. Budget: at most {budget} tool calls. Never repeat an identical call.
11. Never invent timestamps or facts; cite only what tools returned. If tools can't determine it, say so.
12. Pictures: "show me / highlight / colour / mark / segment X, Y and Z (in a picture / as different classes)" ->
   segment_scene with those class words (no time given: the first frame, t = 0). The picture is shown to the user
   automatically; in the answer, say what is highlighted and its coverage, and mention any class that is not
   visible or that the model cannot segment. One segment_scene call is usually enough.
13. Movement: "where do people/cars go / walk / move", "show the paths/routes/trails", "which areas are busiest",
   "where do they stop" -> movement_map (style "heatmap" for busiest areas / where time is spent, else "trails").
   Traffic at a junction or intersection ("traffic flow", "how many turned left", "where do cars come from / go",
   "busiest road") -> junction_flow. Both draw a picture shown to the user; report their numbers as given and say
   that road names / directions are as seen in the picture.
14. Clips: "make a clip / reel / montage", "show me every time X" -> make_highlight_clip. If an earlier tool already
   gave the moments (checkout trips, zone events, junction_flow moments_by_movement, evidence), pass exactly those
   time ranges as segments (e.g. only the turns for "cars turning"); for an object type use
   label (on checkout footage the item type, e.g. label "apple"); use query only when nothing better exists.
   The clip is shown to the user automatically. Write times as m:ss (1:16, not 0:76).
15. Selected object: when the message says the user clicked an object, "this/it/that" means that object.
   What is it / colour / details -> describe_selection. When did it arrive or leave / where did it go / how long
   was it here -> follow_selection. Same as / appears again / where else -> find_selection_elsewhere.

Reply with ONLY one JSON object per turn, either
{{"thought": "brief reasoning", "plan": "(first turn only) 1-2 sentence plan", "action": "call_tool", "tool": "<tool name>", "args": {{...}}}}
or, when you have enough evidence or the budget is spent,
{{"thought": "brief reasoning", "action": "answer"}}"""

FINAL_PROMPT = """Now write the final answer to the user's question: "{question}"
Return ONLY a JSON object:
{{"answer": "direct answer in 1-4 sentences; mention times as m:ss",
  "verdict": "yes" | "no" | "uncertain" | "n/a",
  "confidence": number 0-1,
  "evidence": [{{"t_start": seconds, "t_end": seconds, "reason": "what is visible there that supports the answer"}}],
  "coverage": "which parts of the video were checked and how (e.g. 'All 6 scenes searched; 3 clips watched closely')"}}
Rules: verdict is "n/a" unless the question is a yes/no question about the video; a request ("can you show
me...", "give me...") is not a yes/no question. Prefer "uncertain" over a guess. State the key results
themselves in the answer; never refer the user to "the result" or "the tool output", which they cannot read.
A "yes" needs at least one tool result that clearly shows the event with "visible" not false; if the deciding
detail could not be seen, or results contradicted each other, the verdict is "uncertain" and the answer must say
what could not be determined and why. A "no" must be backed by coverage of the whole video. Evidence times must
come from tool results (max 5 items). Every count or total in the answer must appear in a tool result: copy it,
never re-add numbers yourself."""

GROUNDING_PROMPT = """Your answer contains numbers that do not appear in any tool result: {numbers}.
Rewrite it using only numbers exactly as the tools reported them (e.g. use a tool's total field instead of adding
numbers yourself). If a number cannot be supported, leave it out. Return the same JSON format."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _tool_catalogue() -> str:
    lines = []
    for t in sorted(TOOLS.values(), key=lambda t: (TIER_ORDER.get(t.tier, 9), t.name)):
        props = t.parameters.get("properties", {})
        req = set(t.parameters.get("required", []))
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {t.name}({args}) [{t.tier}] {t.description}")
    return "\n".join(lines)


def _message_dict(m: Message) -> dict:
    return {"id": str(m.id), "role": m.role, "content": m.content, "answer": m.answer, "steps": m.steps or [],
            "selection": m.selection, "created_at": m.created_at.isoformat()}


def _history(conversation_id, exclude_id) -> list[dict]:
    with session_scope() as s:
        msgs = s.scalars(select(Message).where(Message.conversation_id == conversation_id,
                                               Message.id != exclude_id)
                         .order_by(Message.created_at.desc()).limit(8)).all()
    out = []
    for m in reversed(msgs):
        out.append({"role": m.role, "content": m.content[:1500]})
    return out


def _route(message: str) -> str:
    try:
        path = chat_json([{"role": "user", "content": ROUTER_PROMPT.format(message=message)}], agent=True,
                         max_tokens=50).get("path")
    except Exception:
        path = None
    return "general" if path == "general" else "specific"


def _truncate(data: dict, limit: int = 6000) -> str:
    text = json.dumps(data, default=str)
    return text if len(text) <= limit else text[:limit] + '..."(truncated)"'


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


YES_NO_OPENERS = ("did", "does", "do", "is", "are", "was", "were", "has", "have", "had", "can", "could", "will",
                  "would", "should", "is there", "are there")


# "Can you show me..." is a request, not a yes/no question about the video.
_REQUEST = re.compile(r"^(can|could|would|will) you (please )?(show|give|make|draw|highlight|mark|colou?r|segment|"
                      r"generate|create|list|describe|summari[sz]e|explain|count|find|point|narrate|walk)\b")


def is_yes_no(question: str) -> bool:
    q = question.strip().lower()
    if _REQUEST.match(q):
        return False
    return q.startswith(YES_NO_OPENERS) and not q.startswith(("how", "what", "when", "where", "which", "who"))


def _normalise_answer(data: dict, video_id, duration: float, tools_used: list[str], question: str = "") -> dict:
    verdict = str(data.get("verdict") or "n/a").lower()
    if verdict not in {"yes", "no", "uncertain", "n/a"}:
        verdict = "n/a"
    if question and not is_yes_no(question) and verdict in {"yes", "no"}:
        verdict = "n/a"  # counts, times and descriptions don't get a yes/no chip
    confidence = round(min(max(_num(data.get("confidence"), 0.5), 0), 1), 2)
    if verdict == "uncertain":
        confidence = min(confidence, 0.5)  # "uncertain" with high confidence reads as a contradiction
    evidence = []
    for e in (data.get("evidence") or [])[:5]:
        a = min(max(_num(e.get("t_start")), 0), duration or 1e9)
        b = min(max(_num(e.get("t_end"), a), a), duration or 1e9)
        evidence.append({"t_start": round(a, 2), "t_end": round(b, 2),
                         "keyframe_url": nearest_keyframe_url(video_id, (a + b) / 2),
                         "reason": str(e.get("reason") or "")})
    return {"answer": str(data.get("answer") or "").strip() or "I could not determine an answer.",
            "verdict": verdict, "confidence": confidence,
            "evidence": evidence, "coverage": str(data.get("coverage") or ""),
            "tools_used": list(dict.fromkeys(tools_used))}


_TIME = re.compile(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def _numbers(text: str) -> list[float]:
    return [float(n) for n in _NUM.findall(_TIME.sub(" ", text or ""))]


def ungrounded_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers (> 1, not times) in the answer that no tool result or the question contains."""
    allowed: set[float] = set()
    for src in sources:
        for x in _numbers(src):
            allowed.update({x, float(round(x)), float(int(x)), round(x, 1)})
    bad = []
    for x in _numbers(answer):
        if x > 1 and x not in allowed and round(x, 1) not in allowed:
            bad.append(f"{x:g}")
    return list(dict.fromkeys(bad))


def _run_tool(video_id, name: str, args: dict) -> ToolOutput:
    return TOOLS[name].fn(video_id=video_id, **(args or {}))


def _step(name: str, args: dict) -> dict:
    tool = TOOLS[name]
    try:
        label = tool.label(args or {})
    except Exception:
        label = name
    return {"id": uuid.uuid4().hex[:12], "tool": name, "tier": tool.tier, "label": label or name,
            "args": args or {}, "summary": None, "tool_run_id": None, "overlay_url": None, "image_url": None,
            "clip_url": None, "status": "running"}


def answer_message(conversation_id: uuid.UUID, content: str,
                   selection_id: str | None = None) -> Iterator[tuple[str, dict]]:
    """Yields (event_type, payload) pairs for the chat SSE stream."""
    s_ = get_settings()
    with session_scope() as s:
        conv = s.get(Conversation, conversation_id)
        video = s.get(Video, conv.video_id)
        video_id, title, duration = video.id, video.title, video.duration_s or 0
    selection = None
    if selection_id:
        from app.tools.selection import load_selection, selection_public

        try:
            sid, sel_result = load_selection(video_id, selection_id)
            selection = selection_public(sid, sel_result)
        except ValueError:
            selection = None
    with session_scope() as s:
        from app.models import Clip
        n_clips = len(s.scalars(select(Clip.id).where(Clip.video_id == video_id)).all())
        user_msg = Message(conversation_id=conversation_id, role="user", content=content, steps=[],
                           selection=selection)
        s.add(user_msg)
        s.flush()
        user_dict = _message_dict(user_msg)
    yield "user", {"message": user_dict}

    steps: list[dict] = []
    tools_used: list[str] = []

    def finish(answer: dict) -> Iterator[tuple[str, dict]]:
        with session_scope() as s:
            m = Message(conversation_id=conversation_id, role="assistant", content=answer["answer"],
                        answer=answer, steps=steps)
            s.add(m)
            s.flush()
            md = _message_dict(m)
        yield "answer_delta", {"text": answer["answer"]}
        yield "final", {"message": md}

    try:
        path = "specific" if selection else _route(content)
        # ---- Path 1: general analysis ---------------------------------------------------------
        if path == "general":
            yield "plan", {"text": "General analysis: read the video and scene summaries and the timeline, then "
                                   "write a narrated report."}
            for name in ("get_overview", "get_timeline"):
                st = _step(name, {})
                steps.append(st)
                yield "tool_call", {"step": dict(st)}
            report, outputs = build_report(video_id)
            with session_scope() as s:
                s.get(Video, video_id).analysis = report
            for st, (_n, _a, out) in zip(steps, outputs):
                st.update(status="done", summary=out.summary)
                yield "tool_result", {"step": dict(st)}
                tools_used.append(st["tool"])
            moments = "\n".join(f"- {fmt_t(m['t_start'])}-{fmt_t(m['t_end'])}: {m['title']}"
                                for m in report["key_moments"])
            answer = {"answer": report["narration"] + ("\n\nKey moments:\n" + moments if moments else ""),
                      "verdict": "n/a", "confidence": 0.8,
                      "evidence": [{"t_start": m["t_start"], "t_end": m["t_end"],
                                    "keyframe_url": nearest_keyframe_url(video_id, (m["t_start"] + m["t_end"]) / 2),
                                    "reason": m["title"]} for m in report["key_moments"][:5]],
                      "coverage": f"Whole video: all {n_clips} clip captions and scene summaries",
                      "tools_used": tools_used}
            yield from finish(answer)
            return

        # ---- Path 2: specific question --------------------------------------------------------
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            title=title, duration=fmt_t(duration), duration_s=duration, clips=n_clips,
            tools=_tool_catalogue(), budget=s_.agent_max_tool_calls)}]
        messages += _history(conversation_id, user_msg.id)
        if selection:
            from app.tools.selection import context_note

            messages.append({"role": "user", "content": f"{content}\n\n{context_note(selection)}"})
        else:
            messages.append({"role": "user", "content": content})
        seen_calls: set[str] = set()
        sources = [content, f"{duration:.1f}", str(n_clips)]  # where numbers in the answer may come from
        calls = 0
        while calls < s_.agent_max_tool_calls:
            decision = chat_json(messages, agent=True, max_tokens=600)
            log.info("agent decision: %s", json.dumps(decision)[:500])
            if calls == 0 and decision.get("plan"):
                yield "plan", {"text": str(decision["plan"])}
            messages.append({"role": "assistant", "content": json.dumps(decision)})
            action = str(decision.get("action") or "")
            # Models sometimes put the tool name in "action" instead of "call_tool"; accept both.
            name = decision.get("tool") or (action if action in TOOLS else None)
            wants_tool = action == "call_tool" or action in TOOLS or (name in TOOLS and action != "answer")
            if not wants_tool:
                if calls == 0:  # never answer from priors: gather at least one piece of evidence
                    messages.append({"role": "user", "content": "You have not gathered any evidence yet. Call a "
                                                                "tool first (reply with action call_tool)."})
                    calls += 1
                    continue
                break
            args = decision.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if name not in TOOLS:
                messages.append({"role": "user", "content": f"Unknown tool '{name}'. Choose from: {', '.join(TOOLS)}."})
                calls += 1
                continue
            if name in ("describe_selection", "follow_selection", "find_selection_elsewhere"):
                if not selection:
                    messages.append({"role": "user", "content": "No object is selected for this message; use "
                                                                "other tools."})
                    calls += 1
                    continue
                args = {**args, "selection_id": selection["id"]}  # never trust a model-typed id
            key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if key in seen_calls:
                messages.append({"role": "user", "content": "You already made that exact call. Use its result, "
                                                            "try something different, or answer."})
                calls += 1
                continue
            seen_calls.add(key)
            st = _step(name, args)
            steps.append(st)
            yield "tool_call", {"step": dict(st)}
            calls += 1
            try:
                out = _run_tool(video_id, name, args)
                st.update(status="done", summary=out.summary, tool_run_id=out.tool_run_id,
                          overlay_url=out.overlay_url, image_url=out.image_url,
                          clip_url=out.clip_url)
                tools_used.append(name)
                result_text = _truncate(out.data)
                sources += [result_text, out.summary]
            except Exception as e:
                log.exception("tool %s failed", name)
                st.update(status="failed", summary=f"Failed: {e}"[:300])
                result_text = json.dumps({"error": str(e)[:500]})
            yield "tool_result", {"step": dict(st)}
            remaining = s_.agent_max_tool_calls - calls
            messages.append({"role": "user", "content": f"Result of {name}:\n{result_text}\n\n"
                                                        f"({remaining} tool calls left.) Next JSON action:"})

        messages.append({"role": "user", "content": FINAL_PROMPT.format(question=content)})
        final = chat_json(messages, agent=True, max_tokens=900)
        bad = ungrounded_numbers(str(final.get("answer") or ""), sources)
        if bad:  # one rewrite: models re-add or invent totals
            log.info("ungrounded numbers %s; asking for a rewrite", bad)
            messages += [{"role": "assistant", "content": json.dumps(final)},
                         {"role": "user", "content": GROUNDING_PROMPT.format(numbers=", ".join(bad))}]
            final = chat_json(messages, agent=True, max_tokens=900)
        answer = _normalise_answer(final, video_id, duration, tools_used, content)
        if ungrounded_numbers(answer["answer"], sources):
            answer["confidence"] = min(answer["confidence"], 0.4)
            answer["coverage"] = (answer["coverage"] + " Note: some numbers in this answer could not be matched "
                                  "to a tool result.").strip()
        yield from finish(answer)
    except Exception as e:
        log.exception("agent failed")
        yield "error", {"message": f"The agent failed: {e}"}

/** Seconds → "m:ss". Minutes are not wrapped into hours (e.g. 75:03). */
export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** "0:12–0:18", or a single time when the range is under a second. */
export function formatRange(start: number, end: number): string {
  if (Math.floor(end) <= Math.floor(start)) return formatTime(start);
  return `${formatTime(start)}–${formatTime(end)}`;
}

export function formatPercent(fraction: number | null | undefined): string {
  if (fraction == null || !Number.isFinite(fraction)) return "0%";
  return `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatResolution(w: number | null, h: number | null): string | null {
  if (!w || !h) return null;
  return `${w}×${h}`;
}

export function formatRelative(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = (Date.now() - then) / 1000;
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Join class names, skipping falsy values. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Find "m:ss" timestamps in free text, for making them clickable. */
export const TIMESTAMP_RE = /\b(\d{1,3}):([0-5]\d)(?:\.\d+)?\b/g;

export function parseTimestamp(m: string, s: string): number {
  return Number(m) * 60 + Number(s);
}

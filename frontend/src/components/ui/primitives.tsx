import { AlertTriangle, RotateCw } from "lucide-react";
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "../../lib/format";

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={cn("size-3.5 animate-spin", className)}
      aria-hidden="true"
      fill="none"
    >
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-accent-ink hover:brightness-110 active:brightness-95 shadow-[inset_0_-1px_0_rgb(0_0_0/0.15)]",
  secondary: "bg-raised text-ink border border-line hover:border-line-strong hover:bg-surface",
  ghost: "text-ink-2 hover:bg-sunken hover:text-ink",
  danger: "bg-bad text-white hover:brightness-110",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: "sm" | "md";
  icon?: ReactNode;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon, loading, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      disabled={disabled || loading}
      className={cn(
        "inline-flex select-none items-center justify-center gap-1.5 rounded-lg font-medium transition-[background-color,border-color,filter,color] duration-150 disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-7 px-2.5 text-[12.5px]" : "h-9 px-3.5 text-[13.5px]",
        VARIANTS[variant],
        className,
      )}
      {...rest}
    >
      {loading ? <Spinner /> : icon}
      {children}
    </button>
  );
});

export function IconButton({
  label,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex size-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-sunken hover:text-ink disabled:opacity-40",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export function ProgressBar({
  value,
  tone = "accent",
  striped,
  className,
  label,
}: {
  value: number;
  tone?: "accent" | "ok" | "bad" | "mark" | "muted";
  striped?: boolean;
  className?: string;
  label?: string;
}) {
  const fill = {
    accent: "bg-accent",
    ok: "bg-ok",
    bad: "bg-bad",
    mark: "bg-mark",
    muted: "bg-line-strong",
  }[tone];
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-label={label}
      className={cn("h-1.5 overflow-hidden rounded-full bg-sunken", className)}
    >
      <div
        className={cn("h-full rounded-full transition-[width] duration-300 ease-out", fill, striped && "stripes animate-sweep")}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function ErrorNote({
  title,
  message,
  onRetry,
  className,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2.5 rounded-lg border border-bad/30 bg-bad-soft px-3 py-2.5 text-[13px] text-ink",
        className,
      )}
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-bad" aria-hidden />
      <div className="min-w-0 flex-1">
        {title && <p className="font-medium">{title}</p>}
        <p className={cn("break-words", title && "text-ink-2")}>{message}</p>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[12.5px] font-medium text-bad hover:bg-bad/10"
        >
          <RotateCw className="size-3.5" aria-hidden />
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  body,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  body?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center px-6 py-10 text-center", className)}>
      {icon && <div className="mb-3 text-muted">{icon}</div>}
      <p className="font-display text-[17px] font-semibold tracking-[-0.01em] text-ink">{title}</p>
      {body && <div className="mt-1.5 max-w-[34ch] text-[13px] text-muted">{body}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse-soft rounded-md bg-sunken", className)} />;
}

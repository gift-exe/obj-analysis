import type { Tier } from "../api/types";

export const TIER_LABEL: Record<Tier, string> = {
  memory: "Memory",
  vlm: "VLM",
  gpu: "GPU",
  plugin: "Plug-in",
};

export const TIER_STYLE: Record<Tier, string> = {
  memory: "bg-tier-memory-soft text-tier-memory",
  vlm: "bg-tier-vlm-soft text-tier-vlm",
  gpu: "bg-tier-gpu-soft text-tier-gpu",
  plugin: "bg-tier-plugin-soft text-tier-plugin",
};

export const TIER_DOT: Record<Tier, string> = {
  memory: "bg-tier-memory",
  vlm: "bg-tier-vlm",
  gpu: "bg-tier-gpu",
  plugin: "bg-tier-plugin",
};

# obj-analysis frontend

React dashboard for the video-analysis app: library, ingestion progress, player with timeline, and the AI chat panel.
It talks to the backend only through the endpoints in [`../docs/api-contract.md`](../docs/api-contract.md).

## Run

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api` and `/media` to the backend at `http://localhost:8010`.
To point the proxy somewhere else, set `VITE_BACKEND_URL`, e.g. `VITE_BACKEND_URL=http://localhost:9000 npm run dev`.

Other scripts:

| Script | What it does |
| --- | --- |
| `npm run build` | Type-check (`tsc -b`) and build to `dist/` |
| `npm run typecheck` | Type-check only |
| `npm run lint` | ESLint (typescript-eslint + react-hooks) |
| `npm run preview` | Serve the production build (no proxy; put it behind the backend or a reverse proxy) |

## Layout

```
src/
  api/          types.ts (contract types), client.ts (REST + chat stream + ingest EventSource),
                sse.ts (chunk-safe SSE parser), upload.ts (XHR PUT with progress), queries.ts (TanStack Query hooks)
  state/        player.tsx (shared seek / overlay / markers), uploads.tsx, selection.tsx (URL hash), theme.tsx, useChat.ts
  components/
    library/    sidebar, upload + drag-and-drop, video list
    workspace/  header, processing view (8 ingest steps), player, timeline strip
    chat/       analysis report, thread, tool-step rows, answer card, composer
    ui/         small primitives and badges
  lib/          format.ts (m:ss etc.), tiers.ts
```

Colours are CSS variables in `src/index.css` (light and dark), exposed to Tailwind v4 as `bg-surface`, `text-muted`, `bg-tier-gpu` and so on.
The theme toggle sits in the top-left of the library and is remembered in `localStorage`.

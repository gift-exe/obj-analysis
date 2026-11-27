import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const BACKEND = process.env.VITE_BACKEND_URL ?? "http://localhost:8010";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    allowedHosts: ["les-entries-cadillac-classifieds.trycloudflare.com"],
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/media": { target: BACKEND, changeOrigin: true },
    },
  },
});

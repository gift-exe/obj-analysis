import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { SelectionProvider } from "./state/selection";
import { ThemeProvider } from "./state/theme";
import { UploadsProvider } from "./state/uploads";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1, staleTime: 5_000 },
  },
});

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <SelectionProvider>
          <UploadsProvider>
            <App />
          </UploadsProvider>
        </SelectionProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
);

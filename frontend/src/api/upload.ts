import { ApiError } from "./client";

export interface PutFileOptions {
  url: string;
  method: string;
  headers: Record<string, string>;
  file: File;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

/**
 * PUT raw file bytes to a (possibly pre-signed) URL. Uses XMLHttpRequest
 * because fetch has no upload-progress events.
 */
export function putFile({ url, method, headers, file, onProgress, signal }: PutFileOptions): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method || "PUT", url);
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && e.total > 0) onProgress?.(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(1);
        resolve();
      } else {
        reject(new ApiError(xhr.status, `Upload failed (${xhr.status}${xhr.statusText ? ` ${xhr.statusText}` : ""}).`));
      }
    });
    xhr.addEventListener("error", () => reject(new ApiError(0, "Upload failed: the connection was interrupted.")));
    xhr.addEventListener("abort", () => reject(new DOMException("Upload cancelled", "AbortError")));

    if (signal) {
      if (signal.aborted) {
        reject(new DOMException("Upload cancelled", "AbortError"));
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(file);
  });
}

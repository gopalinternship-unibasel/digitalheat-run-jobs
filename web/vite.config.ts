import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The local Python API (python -m dh.serve) runs live demos and photo search on 127.0.0.1:8765.
const api = { target: "http://127.0.0.1:8765", changeOrigin: false };

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": api, "/thumbs": api } },
  preview: { proxy: { "/api": api, "/thumbs": api } },
});

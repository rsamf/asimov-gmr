import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev server proxies API + video traffic to the Flask app on :5001
// (start it with `.venv/bin/python clip_explorer/app.py`).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: {
      "/api": "http://localhost:5001",
      "/video": "http://localhost:5001",
    },
  },
})

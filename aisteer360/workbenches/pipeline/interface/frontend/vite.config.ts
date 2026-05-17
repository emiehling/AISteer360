import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    outDir: resolve(__dirname, "../static/canvas"),
    emptyOutDir: true,
    cssCodeSplit: false,
    sourcemap: false,
    lib: {
      entry: resolve(__dirname, "src/main.tsx"),
      name: "PipelineCanvas",
      formats: ["iife"],
      fileName: () => "pipeline-canvas.js",
    },
    rollupOptions: {
      output: {
        assetFileNames: (asset) => {
          const filename = asset.names?.[0] ?? "";
          if (filename.endsWith(".css")) return "pipeline-canvas.css";
          return "[name][extname]";
        },
      },
    },
  },
});

import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { createHash } from "node:crypto";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, "../static/canvas");

function hashAndManifest(): Plugin {
  return {
    name: "aisteer-hash-and-manifest",
    apply: "build",
    closeBundle() {
      const jsPath = join(OUT_DIR, "pipeline-canvas.js");
      const cssPath = join(OUT_DIR, "pipeline-canvas.css");
      const jsBytes = readFileSync(jsPath);
      const cssBytes = readFileSync(cssPath);
      const jsHash = createHash("sha256").update(jsBytes).digest("hex").slice(0, 10);
      const cssHash = createHash("sha256").update(cssBytes).digest("hex").slice(0, 10);
      const jsName = `pipeline-canvas.${jsHash}.js`;
      const cssName = `pipeline-canvas.${cssHash}.css`;
      renameSync(jsPath, join(OUT_DIR, jsName));
      renameSync(cssPath, join(OUT_DIR, cssName));
      writeFileSync(
        join(OUT_DIR, "manifest.json"),
        JSON.stringify({ js: jsName, css: cssName }, null, 2) + "\n",
      );
    },
  };
}

export default defineConfig({
  plugins: [react(), hashAndManifest()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    outDir: OUT_DIR,
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

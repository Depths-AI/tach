/// <reference types="vite/client" />

import type { BenchmarkReport } from "./benchmarks.js";

declare global {
  var __tachShowcaseReady: Promise<BenchmarkReport>;
}

/// <reference types="vite/client" />

import type { BenchmarkResult } from "./benchmarks.js";

declare global {
  var __tachShowcaseReady: Promise<readonly BenchmarkResult[]>;
}

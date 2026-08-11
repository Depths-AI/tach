#!/usr/bin/env node

import { spawnSync } from "node:child_process";

import { compilerPath } from "./dist/compiler.js";
import { TachError } from "./dist/error.js";

let resolved;
try {
  resolved = await compilerPath();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`tach: ${error instanceof TachError ? `[${error.code}] ` : ""}${message}`);
  process.exit(1);
}

const child = spawnSync(resolved, process.argv.slice(2), {
  stdio: "inherit",
});
if (child.error) {
  console.error(`tach: ${child.error.message}`);
  process.exit(1);
}
if (child.signal) {
  process.kill(process.pid, child.signal);
} else {
  process.exit(child.status ?? 1);
}

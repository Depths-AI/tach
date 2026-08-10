#!/usr/bin/env node

import { spawnSync } from "node:child_process";

import { compilerPath } from "./dist/compiler.js";

const resolved = await compilerPath();
if (!resolved.ok) {
  console.error(`tach: [${resolved.error.code}] ${resolved.error.message}`);
  process.exit(1);
}

const child = spawnSync(resolved.value, process.argv.slice(2), {
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

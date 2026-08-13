#!/usr/bin/env node

import { spawnSync } from "node:child_process";

import { compilerPath, runCompiler } from "./dist/compiler.js";
import { TachError } from "./dist/error.js";

const args = process.argv.slice(2);
const help = args[0] === "help" || args[0] === "-h" || args[0] === "--help";
if (args[0] === "docs" || help) {
  try {
    const result = await runCompiler(args);
    const output = help
      ? result.stdout
        .replace(
          "  tach check [--target web|spirv|all] FILE.tach\n",
          "$&  tach docs FILE.tach\n",
        )
        .replace(
          "  check      validate the WebGPU pipeline by default; use --target for SPIR-V or all\n",
          "$&  docs       generate Markdown API documentation on standard output\n",
        )
      : result.stdout;
    process.stdout.write(output);
    process.stderr.write(result.stderr);
    process.exit(0);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`tach: ${error instanceof TachError ? `[${error.code}] ` : ""}${message}`);
    process.exit(1);
  }
}

let resolved;
try {
  resolved = await compilerPath();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`tach: ${error instanceof TachError ? `[${error.code}] ` : ""}${message}`);
  process.exit(1);
}

const child = spawnSync(resolved, args, {
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

import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "@depths/tach/compiler";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(projectRoot, "kernels", "benchmarks.tach");
const result = await build(source, { cwd: projectRoot });
if (!result.ok) {
  throw new Error(`[${result.error.code}] ${result.error.message}`, {
    cause: result.error.cause,
  });
}
if (result.value.stdout) process.stdout.write(result.value.stdout);
if (result.value.stderr) process.stderr.write(result.value.stderr);

await Promise.all([
  access(join(projectRoot, "build", "benchmarks.js"), constants.R_OK),
  access(join(projectRoot, "build", "benchmarks.d.ts"), constants.R_OK),
]);

import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "@depths/tach/compiler";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(projectRoot, "kernels", "benchmarks.tach");
const result = await build(source, { cwd: projectRoot });
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);

await Promise.all([
  access(join(projectRoot, "build", "benchmarks.js"), constants.R_OK),
  access(join(projectRoot, "build", "benchmarks.d.ts"), constants.R_OK),
]);

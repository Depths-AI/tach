import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "@depths/tach/compiler";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const kernels = ["procedural", "mesh", "matrix", "monte-carlo", "particles", "wave"];

for (const kernel of kernels) {
  const result = await build(join(root, "kernels", `${kernel}.tach`), { cwd: root });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  await Promise.all(["js", "d.ts", "wgsl"].map((extension) =>
    access(join(root, "build", `${kernel}.${extension}`), constants.R_OK)));
}

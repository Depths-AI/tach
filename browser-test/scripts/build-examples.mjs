import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "@depths/tach/compiler";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

export default async function buildExamples() {
  await build({ cwd: resolve(root, "examples") });
}

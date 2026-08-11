import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compilerPath } from "../dist/compiler.js";

try {
  console.log(`@depths/tach compiler: ${await compilerPath()}`);
} catch (error) {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const info = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));
  const message = error instanceof Error ? error.message : String(error);
  if (info.version === "0.0.0") {
    console.warn(`@depths/tach: ${message}`);
  } else {
    console.error(`@depths/tach: ${message}`);
    process.exitCode = 1;
  }
}

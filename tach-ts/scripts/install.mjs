import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compilerPath } from "../dist/compiler.js";

const result = await compilerPath();
if (result.ok) {
  console.log(`@depths/tach compiler: ${result.value}`);
} else {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const info = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));
  if (info.version === "0.0.0") {
    console.warn(`@depths/tach: ${result.error.message}`);
  } else {
    console.error(`@depths/tach: ${result.error.message}`);
    process.exitCode = 1;
  }
}

import { mkdir, readdir, rm } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build, runCompiler } from "@depths/tach/compiler";

const scriptPath = fileURLToPath(import.meta.url);
const projectRoot = resolve(dirname(scriptPath), "..");
const repositoryRoot = resolve(projectRoot, "..");
const examplesDir = join(repositoryRoot, "examples");
const outputDir = join(projectRoot, "build");
function print(result) {
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  return result;
}

export default async function buildExamples() {
  print(await runCompiler(["version"], { cwd: projectRoot }));
  const sources = (await readdir(examplesDir)).filter((name) => name.endsWith(".tach")).sort();
  if (sources.length === 0) throw new Error(`no Tach examples found in ${examplesDir}`);

  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });

  for (const sourceName of sources) {
    const sourcePath = join(examplesDir, sourceName);
    console.log(`Compiling ${relative(repositoryRoot, sourcePath)}`);
    print(await build(sourcePath, { cwd: projectRoot }));

  }
}

if (process.argv[1] === scriptPath) await buildExamples();

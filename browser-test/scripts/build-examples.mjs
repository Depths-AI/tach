import { access, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(projectRoot, "..");
const examplesDir = join(repositoryRoot, "examples");
const outputDir = join(projectRoot, "build");
const executableName = process.platform === "win32" ? "tach.exe" : "tach";

async function findTach() {
  const candidates = [];
  if (process.env.TACH_BIN) {
    candidates.push(isAbsolute(process.env.TACH_BIN) ? process.env.TACH_BIN : resolve(process.cwd(), process.env.TACH_BIN));
  }
  candidates.push(join(repositoryRoot, "bin", executableName));
  if (process.platform === "win32") candidates.push(join(repositoryRoot, "bin", "tach"));

  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Try the next explicit candidate before falling back to PATH.
    }
  }
  return executableName;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    encoding: "utf8",
    ...options,
  });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} exited with status ${result.status}`);
  return result.stdout.trim();
}

const tach = await findTach();
const cliVersion = run(tach, ["version"]);
const sources = (await readdir(examplesDir)).filter((name) => name.endsWith(".tach")).sort();
if (sources.length === 0) throw new Error(`no Tach examples found in ${examplesDir}`);

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });

const artifactSuffixes = [".tir", ".wgsl", ".spv", ".spvasm", ".js", ".d.ts", ".tach.json"];
const examples = [];

for (const sourceName of sources) {
  const sourcePath = join(examplesDir, sourceName);
  const name = sourceName.slice(0, -".tach".length);
  console.log(`Compiling ${relative(repositoryRoot, sourcePath)}`);
  run(tach, ["build", sourcePath]);

  for (const suffix of artifactSuffixes) await access(join(outputDir, name + suffix), constants.R_OK);
  const metadata = JSON.parse(await readFile(join(outputDir, `${name}.tach.json`), "utf8"));
  examples.push({
    name,
    source: relative(repositoryRoot, sourcePath).replaceAll("\\", "/"),
    module: `/build/${name}.js`,
    metadata: `/build/${name}.tach.json`,
    kernels: metadata.kernels.map((kernel) => kernel.name),
  });
}

await writeFile(
  join(outputDir, "manifest.json"),
  JSON.stringify({ cliVersion, generatedAt: new Date().toISOString(), examples }, null, 2) + "\n",
);
console.log(`Prepared ${examples.length} browser fixtures in ${relative(repositoryRoot, outputDir)}`);

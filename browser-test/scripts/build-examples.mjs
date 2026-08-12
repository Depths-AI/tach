import { access, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
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
  const cliVersion = print(await runCompiler(["version"], { cwd: projectRoot })).stdout.trim();
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
    print(await build(sourcePath, { cwd: projectRoot, target: "all" }));

    for (const suffix of artifactSuffixes) await access(join(outputDir, name + suffix), constants.R_OK);
    const metadata = JSON.parse(await readFile(join(outputDir, `${name}.tach.json`), "utf8"));
		if (metadata.schema !== 1 || !metadata.targets?.web) throw new Error(`${name} has invalid schema-1 Web metadata`);
    const wgsl = await readFile(join(outputDir, `${name}.wgsl`), "utf8");
		const plans = metadata.targets.web.programs;
    examples.push({
      name,
      source: relative(repositoryRoot, sourcePath).replaceAll("\\", "/"),
      module: `/build/${name}.js`,
      metadata: `/build/${name}.tach.json`,
			programs: metadata.programs.map((program) => program.name),
			physicalKernels: metadata.targets.web.kernels.length,
			dispatches: plans.reduce((count, plan) => count + plan.steps.filter((step) => step.kind === "dispatch").length, 0),
			transients: plans.reduce((count, plan) => count + plan.transients.length, 0),
      wgslBytes: Buffer.byteLength(wgsl),
    });
  }

  await writeFile(
    join(outputDir, "manifest.json"),
    JSON.stringify({ cliVersion, generatedAt: new Date().toISOString(), examples }, null, 2) + "\n",
  );
  console.log(`Prepared ${examples.length} browser fixtures in ${relative(repositoryRoot, outputDir)}`);
}

if (process.argv[1] === scriptPath) await buildExamples();

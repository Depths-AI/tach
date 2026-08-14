import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";

import { build, check, compilerPath, docs, format } from "../dist/compiler.js";

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const cli = join(packageRoot, "cli.mjs");
const manifest = {
  name: "fixture",
  version: "0.1.0",
  web: { package: "@test/fixture" },
  docs: { title: "Fixture kernels", summary: "Typed fixture kernels." },
};

async function fixture(source, parent = tmpdir()) {
  const root = await mkdtemp(join(parent, "tach-project-"));
  await mkdir(join(root, "kernels"));
  await writeFile(join(root, "tach.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(join(root, "kernels", "scale.tach"), source);
  return root;
}

const source = `
@docs(title("Scaling"), summary("Scales numeric values."));
@docs(summary("A scale configuration."), field(factor, "Multiplier."))
type Options = { factor: float32 };
type ComputeBuffer = { value: float32 };
@docs(summary("Gets the multiplier."), param(options, "Scale configuration."), returns("The configured multiplier."))
function multiplier(options: Options): float32 { return options.factor; }
@docs(summary("Scales every value."), coordinate(i, "Value index."), param(values, "Values to update."), param(options, "Scaling options."))
export function scale[i](values: buffer<float32[]>, options: Options) {
  if (i < values.length) { values[i] *= multiplier(options); }
}
export function tach[i](gpu: buffer<ComputeBuffer[]>) {
  if (i < gpu.length) { gpu[i].value *= 2.0; }
}
`;

test("the public CLI is authoritative and the private compiler resolves", async () => {
  await access(await compilerPath());
  const helps = ["help", "-h", "--help"].map((argument) => spawnSync(process.execPath, [cli, argument], { encoding: "utf8" }));
  for (const help of helps) {
    assert.equal(help.status, 0, help.stderr);
    assert.match(help.stdout, /tach build \[--target web\|spirv\]/u);
    assert.match(help.stdout, /tach instructions \[--details <section>\.\.\.\]/u);
    assert.equal(help.stdout, helps[0].stdout);
  }
  const version = spawnSync(process.execPath, [cli, "version"], { encoding: "utf8" });
  assert.equal(version.status, 0, version.stderr);
  assert.match(version.stdout, /^tach /u);
  const unknown = spawnSync(process.execPath, [cli, "unknown"], { encoding: "utf8" });
  assert.notEqual(unknown.status, 0);
  assert.match(unknown.stderr, /unknown command/u);
  for (const args of [["build", "kernel.tach"], ["build", "--target", "invalid"], ["check", "--target", "web"], ["fmt", "kernel.tach"]]) {
    const rejected = spawnSync(process.execPath, [cli, ...args], { encoding: "utf8" });
    assert.notEqual(rejected.status, 0, `${args.join(" ")} unexpectedly succeeded`);
  }
});

test("bundled instructions expose dense context and exact detail chunks", async () => {
  const bundle = JSON.parse(await readFile(join(packageRoot, "dist", "instructions.json"), "utf8"));
  assert.equal(bundle.schema, 1);
  assert.deepEqual(Object.keys(bundle.sections), Array.from({ length: 85 }, (_, index) => String(index + 1)));

  const mini = spawnSync(process.execPath, [cli, "instructions"], { cwd: tmpdir(), encoding: "utf8" });
  assert.equal(mini.status, 0, mini.stderr);
  assert.equal(mini.stdout.trim(), bundle.mini);
  assert.ok(mini.stdout.trim().split(/\s+/u).length >= 1_500);
  assert.doesNotMatch(mini.stdout, /INSTRUCTIONS\.md#/u);

  const details = spawnSync(process.execPath, [cli, "instructions", "--details", "47", "6", "47"], { cwd: tmpdir(), encoding: "utf8" });
  assert.equal(details.status, 0, details.stderr);
  assert.ok(details.stdout.indexOf("## 47. CLI command surface") < details.stdout.indexOf("## 6. Imports"));
  assert.equal(details.stdout.match(/^## 47\./gmu)?.length, 1);
  assert.doesNotMatch(details.stdout, /^## 7\./mu);

  for (const args of [["instructions", "--details"], ["instructions", "--details", "0"], ["instructions", "--details", "86"], ["instructions", "--details", "1,2"], ["instructions", "unexpected"]]) {
    const rejected = spawnSync(process.execPath, [cli, ...args], { cwd: tmpdir(), encoding: "utf8" });
    assert.notEqual(rejected.status, 0, `${args.join(" ")} unexpectedly succeeded`);
  }
});

test("web and SPIR-V builds replace the fixed artifact tree", async () => {
  const root = await fixture(source);
  try {
    await build({ cwd: root });
    assert.deepEqual((await readdir(join(root, "build"))).sort(), ["README.md", "docs", "index.d.ts", "index.js", "kernel.wgsl", "package.json"]);
    assert.deepEqual(await readdir(join(root, "build", "docs")), ["kernels.md"]);
    const generated = await readFile(join(root, "build", "index.js"), "utf8");
    assert.match(generated, /from "@depths\/tach\/internal"/u);
    assert.match(generated, /export function scale/u);
    assert.doesNotMatch(generated, /export\s*\{[^}]*\btach\b|from "@depths\/tach";/u);
    const declarations = await readFile(join(root, "build", "index.d.ts"), "utf8");
    assert.match(declarations, /export type Options/u);
    assert.match(declarations, /export type ComputeBuffer/u);
    assert.match(declarations, /export function scale/u);
    assert.match(declarations, /export function tach/u);
    assert.match(declarations, /ComputeBuffer as \$ComputeBuffer/u);
    assert.doesNotMatch(declarations, /export function multiplier/u);
    const generatedPackage = JSON.parse(await readFile(join(root, "build", "package.json"), "utf8"));
    assert.deepEqual(generatedPackage, {
      name: "@test/fixture",
      version: "0.1.0",
      type: "module",
      sideEffects: false,
      exports: { ".": { types: "./index.d.ts", import: "./index.js" } },
      dependencies: { "@depths/tach": "0.0.0" },
    });

    await build({ cwd: join(root, "kernels"), target: "spirv" });
    assert.deepEqual((await readdir(join(root, "build"))).sort(), ["README.md", "docs", "kernel.spv"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("check writes nothing and a failed build preserves complete output", async () => {
  const root = await fixture(source);
  try {
    await build({ cwd: root });
    const before = await readFile(join(root, "build", "index.js"));
    const inventory = (await readdir(join(root, "build"))).sort();
    await check({ cwd: root });
    assert.deepEqual((await readdir(join(root, "build"))).sort(), inventory);
    assert.deepEqual(await readFile(join(root, "build", "index.js")), before);
    await writeFile(join(root, "kernels", "scale.tach"), "export function broken[");
    await assert.rejects(build({ cwd: root }));
    assert.deepEqual((await readdir(join(root, "build"))).sort(), inventory);
    assert.deepEqual(await readFile(join(root, "build", "index.js")), before);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("docs changes only documentation and formatter is project-wide", async () => {
  const root = await fixture(source);
  try {
    await build({ cwd: root });
    const preserved = new Map(await Promise.all(
      ["index.d.ts", "index.js", "kernel.wgsl", "package.json"].map(async (name) => [name, await readFile(join(root, "build", name))]),
    ));
    await writeFile(join(root, "build", "docs", "stale.md"), "stale\n");
    await writeFile(join(root, "tach.json"), `${JSON.stringify({ ...manifest, docs: { ...manifest.docs, summary: "Updated summary." } }, null, 2)}\n`);
    await docs({ cwd: root });
    assert.match(await readFile(join(root, "build", "README.md"), "utf8"), /Updated summary\./u);
    assert.deepEqual(await readdir(join(root, "build", "docs")), ["kernels.md"]);
    for (const [name, contents] of preserved) assert.deepEqual(await readFile(join(root, "build", name)), contents);

    const kernel = join(root, "kernels", "scale.tach");
    const readme = await readFile(join(root, "build", "README.md"));
    await writeFile(kernel, `@docs(title("Missing summary"));\n${source}`);
    await assert.rejects(docs({ cwd: root }));
    assert.deepEqual(await readFile(join(root, "build", "README.md")), readme);
    for (const [name, contents] of preserved) assert.deepEqual(await readFile(join(root, "build", name)), contents);
    await writeFile(kernel, `// preserved\n${source.replaceAll("  ", "    ")}`);
    await format({ cwd: join(root, "kernels") });
    const once = await readFile(kernel, "utf8");
    assert.match(once, /\/\/ preserved/u);
    await format({ cwd: root });
    assert.equal(await readFile(kernel, "utf8"), once);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("docs creates only documentation before the first build", async () => {
  const root = await fixture(source);
  try {
    await docs({ cwd: root });
    assert.deepEqual((await readdir(join(root, "build"))).sort(), ["README.md", "docs"]);
    assert.deepEqual(await readdir(join(root, "build", "docs")), ["kernels.md"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("generated documentation usage type-checks against the package facade", async () => {
  const root = await fixture(source, dirname(fileURLToPath(import.meta.url)));
  try {
    await mkdir(join(root, "data"));
    await writeFile(join(root, "data", "extra.tach"), `
@docs(title("Shared data"), summary("Additional documented types."));
@docs(summary("An extra host value."), field(value, "Magnitude | normalized."))
type Extra = { value: float32 };
`);
    await build({ cwd: root });
    const readme = await readFile(join(root, "build", "README.md"), "utf8");
    assert.match(readme, /^# Fixture kernels/mu);
    assert.ok(readme.indexOf("docs/data.md") < readme.indexOf("docs/kernels.md"));
    assert.deepEqual((await readdir(join(root, "build", "docs"))).sort(), ["data.md", "kernels.md"]);
    assert.match(await readFile(join(root, "build", "docs", "data.md"), "utf8"), /## Shared data[\s\S]*Magnitude \\| normalized\./u);
    assert.match(await readFile(join(root, "build", "docs", "kernels.md"), "utf8"), /Internal functions and stages[\s\S]*TypeScript-callable programs/u);
    const snippet = readme.match(/```ts\n([\s\S]*?)```/u)?.[1];
    assert.ok(snippet);
    const example = join(root, "build", "usage.ts");
    await writeFile(example, snippet);
    const tsc = join(packageRoot, "..", "node_modules", "typescript", "lib", "tsc.js");
    const checked = spawnSync(process.execPath, [tsc, "--ignoreConfig", "--noEmit", "--strict", "--skipLibCheck", "--target", "ES2022", "--module", "NodeNext", "--moduleResolution", "NodeNext", example], { cwd: join(root, "build"), encoding: "utf8" });
    assert.equal(checked.status, 0, checked.stdout + checked.stderr);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("TACH_BIN must name an executable file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "tach-bin-test-"));
  const previous = process.env.TACH_BIN;
  process.env.TACH_BIN = directory;
  try {
    await assert.rejects(compilerPath(), (error) => {
      assert.equal(error.code, "compiler-install");
      return true;
    });
  } finally {
    if (previous === undefined) delete process.env.TACH_BIN;
    else process.env.TACH_BIN = previous;
    await rm(directory, { recursive: true });
  }
});

import assert from "node:assert/strict";
import { access, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";

import { build, compilerPath, runCompiler } from "../dist/compiler.js";

const cli = join(dirname(fileURLToPath(import.meta.url)), "..", "cli.mjs");

test("the package resolves and runs the repository compiler", async () => {
  const path = await compilerPath();
  await access(path);

  const version = await runCompiler(["version"]);
  assert.match(version.stdout, /^tach /u);
  const help = spawnSync(process.execPath, [cli, "help"], { encoding: "utf8" });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /tach docs FILE\.tach/u);
});

test("build defaults to web artifacts and can select SPIR-V", async () => {
  const directory = await mkdtemp(join(tmpdir(), "tach-package-test-"));
  try {
    const source = join(directory, "scale.tach");
    await writeFile(source, `
@workgroup(1)
export function scale[i](data: buffer<float32[]>, factor: float32) {
  if (i < data.length) { data[i] *= factor; }
}
`);
    await build(source, { cwd: directory });
    const generated = await readFile(join(directory, "build", "scale.js"), "utf8");
    assert.match(generated, /from "@depths\/tach\/internal"/u);
		assert.match(generated, /export function scale\(\.\.\.\$args\)/u);
    assert.doesNotMatch(generated, /export const buffer/u);
    assert.deepEqual((await readdir(join(directory, "build"))).sort(), ["scale.d.ts", "scale.js", "scale.wgsl"]);

    const spirv = await build(source, { cwd: directory, target: "spirv" });
    assert.equal(spirv.stderr, "");
		assert.deepEqual((await readdir(join(directory, "build"))).sort(), ["scale.spv"]);

    const checked = await runCompiler(["check", source], { cwd: directory });
		assert.match(checked.stdout, /WGSL:.*programs:/su);
    assert.doesNotMatch(checked.stdout, /SPIR-V:/u);

    const checkedSPIRV = await runCompiler(["check", "--target", "spirv", source], { cwd: directory });
    assert.match(checkedSPIRV.stdout, /SPIR-V:/u);
		assert.doesNotMatch(checkedSPIRV.stdout, /WGSL:|programs:/u);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("compiler process failures reject with a typed error", async () => {
  await assert.rejects(runCompiler(["definitely-not-a-command"]), (error) => {
    assert.equal(error.code, "compiler-execution");
    assert.match(error.message, /exit code/u);
    return true;
  });
});

test("compiler setup failures reject with a typed error", async () => {
  await assert.rejects(runCompiler(["version"], { cwd: 42 }), (error) => {
    assert.equal(error.code, "compiler-execution");
    assert.equal(error.operation, "compiler");
    return true;
  });
});

test("TACH_BIN must name an executable file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "tach-bin-test-"));
  const previous = process.env.TACH_BIN;
  process.env.TACH_BIN = directory;
  try {
    await assert.rejects(compilerPath(), (error) => {
      assert.equal(error.code, "compiler-install");
      assert.match(error.message, /does not point to an executable/u);
      return true;
    });
  } finally {
    if (previous === undefined) delete process.env.TACH_BIN;
    else process.env.TACH_BIN = previous;
    await rm(directory, { recursive: true });
  }
});

test("docs renders and type-checks usage from validated Tach semantics", async () => {
  const directory = await mkdtemp(join(dirname(fileURLToPath(import.meta.url)), "docs-"));
  try {
    const source = join(directory, "scale.tach");
    await writeFile(source, `
@docs(title("Scaling kernels"), summary("Typed GPU scaling."));
@docs(summary("A scale configuration."), field(factor, "Multiplier."))
type Options = { factor: float32 };
@docs(summary("Gets the multiplier."), param(options, "Scale configuration."), returns("The configured multiplier."))
function multiplier(options: Options): float32 { return options.factor; }
@docs(summary("Scales every value."), coordinate(i, "Value index."), param(values, "Values to update."), param(options, "Scaling options."))
export function scale[i](values: buffer<float32[]>, options: Options) {
  if (i < values.length) { values[i] *= multiplier(options); }
}
`);
    await build(source, { cwd: directory });
    const generated = spawnSync(process.execPath, [cli, "docs", source], { cwd: directory, encoding: "utf8" });
    assert.equal(generated.status, 0, generated.stderr);
    assert.match(generated.stdout, /^# Scaling kernels/mu);
    assert.match(generated.stdout, /GPU buffer · read\/write/u);
    assert.match(generated.stdout, /## Internal functions and stages[\s\S]*The configured multiplier\./u);
    assert.match(generated.stdout, /## Exported programs[\s\S]*Scales every value\./u);
    assert.match(generated.stdout, /## TypeScript usage/u);
    assert.match(generated.stdout, /\$size: number[\s\S]*\{ size: \$size \}/u);
    const snippet = generated.stdout.match(/```ts\n([\s\S]*?)```/u)?.[1];
    assert.ok(snippet);
    const example = join(directory, "usage.ts");
    await writeFile(example, snippet);
    const tsc = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "node_modules", "typescript", "lib", "tsc.js");
    const checked = spawnSync(process.execPath, [tsc, "--ignoreConfig", "--noEmit", "--strict", "--skipLibCheck", "--target", "ES2022", "--module", "NodeNext", "--moduleResolution", "NodeNext", example], { cwd: directory, encoding: "utf8" });
    assert.equal(checked.status, 0, checked.stdout + checked.stderr);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

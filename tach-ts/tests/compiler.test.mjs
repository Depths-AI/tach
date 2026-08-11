import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { build, compilerPath, runCompiler } from "../dist/compiler.js";

test("the package resolves and runs the repository compiler", async () => {
  const path = await compilerPath();
  assert.equal(path.ok, true);
  await access(path.value);

  const version = await runCompiler(["version"]);
  assert.equal(version.ok, true);
  assert.match(version.value.stdout, /^tach /u);
});

test("build emits package-backed JavaScript bindings", async () => {
  const directory = await mkdtemp(join(tmpdir(), "tach-package-test-"));
  try {
    const source = join(directory, "scale.tach");
    await writeFile(source, `
@workgroup(1)
export function scale[i](data: buffer<float32[]>, factor: float32) {
  if (i < data.length) { data[i] *= factor; }
}
`);
    const result = await build(source, { cwd: directory });
    assert.equal(result.ok, true);
    const generated = await readFile(join(directory, "build", "scale.js"), "utf8");
    assert.match(generated, /from "@depths\/tach\/internal"/u);
    assert.match(generated, /export function scale\(data, factor, \$dispatch\)/u);
    assert.doesNotMatch(generated, /export const buffer/u);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("compiler process failures are returned as data", async () => {
  const result = await runCompiler(["definitely-not-a-command"]);
  assert.equal(result.ok, false);
  assert.equal(result.error.code, "compiler-execution");
  assert.match(result.error.message, /exit code/u);
});

test("compiler setup failures are returned as data", async () => {
  const result = await runCompiler(["version"], { cwd: 42 });
  assert.equal(result.ok, false);
  assert.equal(result.error.code, "compiler-execution");
  assert.equal(result.error.operation, "compiler");
});

test("TACH_BIN must name an executable file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "tach-bin-test-"));
  const previous = process.env.TACH_BIN;
  process.env.TACH_BIN = directory;
  try {
    const result = await compilerPath();
    assert.equal(result.ok, false);
    assert.equal(result.error.code, "compiler-install");
    assert.match(result.error.message, /does not point to an executable/u);
  } finally {
    if (previous === undefined) delete process.env.TACH_BIN;
    else process.env.TACH_BIN = previous;
    await rm(directory, { recursive: true });
  }
});

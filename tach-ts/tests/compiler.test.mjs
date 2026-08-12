import assert from "node:assert/strict";
import { access, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { build, compilerPath, runCompiler } from "../dist/compiler.js";

test("the package resolves and runs the repository compiler", async () => {
  const path = await compilerPath();
  await access(path);

  const version = await runCompiler(["version"]);
  assert.match(version.stdout, /^tach /u);
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

import { assert } from "./assert.js";
import { build, check, compilerPath, docs, format } from "../dist/compiler.js";

const packageRoot = decodeURIComponent(new URL("../", import.meta.url).pathname)
  .replace(/^\/(.:\/)/u, "$1");
const cli = `${packageRoot}/src/cli.ts`;
const manifest = {
  name: "fixture",
  version: "0.1.0",
  javascript: { package: "@test/fixture" },
  docs: { title: "Fixture kernels", summary: "Typed fixture kernels." },
};
const source = `
@docs(title("Scaling"), summary("Scales numeric values."));
@docs(summary("A scale configuration."), field(factor, "Multiplier."))
type Options = { factor: float32 };
type ComputeBuffer = { value: float32 };
type Half = { value: float16 };
@docs(summary("Gets the multiplier."), param(options, "Scale configuration."), returns("The configured multiplier."))
function multiplier(options: Options): float32 { return options.factor; }
@docs(summary("Scales every value."), coordinate(i, "Value index."), param(values, "Values to update."), param(options, "Scaling options."))
export function scale[i](values: buffer<float32[]>, options: Options) {
  if (i < values.length) { values[i] *= multiplier(options); }
}
export function tach[i](gpu: buffer<ComputeBuffer[]>) {
  if (i < gpu.length) { gpu[i].value *= 2.0; }
}
export function half[i](values: buffer<float16[]>, factor: float16) {
  if (i < values.length) { values[i] *= factor; }
}
`;

const join = (root, ...parts) =>
  [root.replace(/[\\/]+$/u, ""), ...parts].join("/");
async function names(path) {
  const result = [];
  for await (const entry of Deno.readDir(path)) result.push(entry.name);
  return result.sort();
}
async function fixture(kernel, directory) {
  const root = await Deno.makeTempDir({
    ...(directory ? { dir: directory } : {}),
    prefix: "tach-project-",
  });
  await Deno.mkdir(join(root, "kernels"));
  await Deno.writeTextFile(
    join(root, "tach.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  await Deno.writeTextFile(join(root, "kernels", "scale.tach"), kernel);
  return root;
}
function command(program, args, cwd) {
  const result = new Deno.Command(program, {
    args,
    ...(cwd ? { cwd } : {}),
    stdout: "piped",
    stderr: "piped",
  }).outputSync();
  return {
    status: result.code,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}
const runCLI = (args, cwd) =>
  command(Deno.execPath(), ["run", "-A", cli, ...args], cwd);

Deno.test("the public CLI is authoritative and the private compiler resolves", async () => {
  assert.ok((await Deno.stat(await compilerPath())).isFile);
  const helps = ["help", "-h", "--help"].map((argument) => runCLI([argument]));
  for (const help of helps) {
    assert.equal(help.status, 0, help.stderr);
    assert.match(help.stdout, /tach build \[--verbose\]/u);
    assert.match(
      help.stdout,
      /tach instructions \[--details <section>\.\.\.\]/u,
    );
    assert.equal(help.stdout, helps[0].stdout);
  }
  const version = runCLI(["version"]);
  assert.equal(version.status, 0, version.stderr);
  assert.match(version.stdout, /^tach /u);
  const unknown = runCLI(["unknown"]);
  assert.notEqual(unknown.status, 0);
  assert.match(unknown.stderr, /unknown command/u);
  for (
    const args of [
      ["build", "kernel.tach"],
      ["build", "--unknown"],
      ["build", "--verbose", "extra"],
      ["check", "--verbose"],
      ["fmt", "kernel.tach"],
    ]
  ) {
    assert.notEqual(
      runCLI(args).status,
      0,
      `${args.join(" ")} unexpectedly succeeded`,
    );
  }
});

Deno.test("bundled instructions expose dense context and exact detail chunks", async () => {
  const bundle = JSON.parse(
    await Deno.readTextFile(join(packageRoot, "dist", "instructions.json")),
  );
  assert.equal(bundle.schema, 1);
  assert.deepEqual(
    Object.keys(bundle.sections),
    Array.from({ length: 85 }, (_, index) => String(index + 1)),
  );
  const mini = runCLI(["instructions"], await Deno.makeTempDir());
  assert.equal(mini.status, 0, mini.stderr);
  assert.equal(mini.stdout.trim(), bundle.mini);
  assert.ok(mini.stdout.trim().split(/\s+/u).length >= 1_500);
  assert.doesNotMatch(mini.stdout, /INSTRUCTIONS\.md#/u);
  const details = runCLI(["instructions", "--details", "47", "6", "47"]);
  assert.equal(details.status, 0, details.stderr);
  assert.ok(
    details.stdout.indexOf("## 47. CLI command surface") <
      details.stdout.indexOf("## 6. Imports"),
  );
  assert.equal(details.stdout.match(/^## 47\./gmu)?.length, 1);
  assert.doesNotMatch(details.stdout, /^## 7\./mu);
  for (
    const args of [
      ["instructions", "--details"],
      ["instructions", "--details", "0"],
      ["instructions", "--details", "86"],
      ["instructions", "--details", "1,2"],
      ["instructions", "unexpected"],
    ]
  ) {
    assert.notEqual(
      runCLI(args).status,
      0,
      `${args.join(" ")} unexpectedly succeeded`,
    );
  }
});

Deno.test("one build emits the complete dual-backend package", async () => {
  const root = await fixture(source);
  try {
    await build({ cwd: root });
    assert.deepEqual(await names(join(root, "build")), [
      "README.md",
      "docs",
      "index.d.ts",
      "index.js",
      "kernel.spv",
      "kernel.wgsl.gz",
      "package.json",
    ]);
    assert.deepEqual(await names(join(root, "build", "docs")), ["kernels.md"]);
    const generated = await Deno.readTextFile(join(root, "build", "index.js"));
    assert.match(generated, /from "@depths\/tach\/internal"/u);
    assert.match(generated, /export function scale/u);
    assert.doesNotMatch(
      generated,
      /export\s*\{[^}]*\btach\b|from "@depths\/tach";/u,
    );
    assert.match(
      generated,
      /new URL\("\.\/kernel\.wgsl\.gz\?v=[0-9a-f]{64}", import\.meta\.url\)/u,
    );
    assert.match(generated, /new URL\("\.\/kernel\.spv", import\.meta\.url\)/u);
    const declarations = await Deno.readTextFile(
      join(root, "build", "index.d.ts"),
    );
    for (
      const pattern of [
        /export type Options/u,
        /export type ComputeBuffer/u,
        /export type Half/u,
        /export function scale/u,
        /export function tach/u,
        /export function half/u,
        /ComputeBuffer<Float16Array \| readonly number\[\]>/u,
        /ComputeBuffer as \$ComputeBuffer/u,
      ]
    ) assert.match(declarations, pattern);
    assert.doesNotMatch(declarations, /export function multiplier/u);
    assert.deepEqual(
      JSON.parse(await Deno.readTextFile(join(root, "build", "package.json"))),
      {
        name: "@test/fixture",
        version: "0.1.0",
        type: "module",
        sideEffects: false,
        exports: { ".": { types: "./index.d.ts", default: "./index.js" } },
        dependencies: { "@depths/tach": "0.1.3" },
      },
    );
    const artifacts = [
      "index.d.ts",
      "index.js",
      "kernel.spv",
      "kernel.wgsl.gz",
      "package.json",
    ];
    const ordinary = new Map(
      await Promise.all(
        artifacts.map(async (
          name,
        ) => [name, await Deno.readFile(join(root, "build", name))]),
      ),
    );
    await build({ cwd: join(root, "kernels"), verbose: true });
    assert.deepEqual(await names(join(root, "build")), [
      "README.md",
      "diagnostics",
      "docs",
      "index.d.ts",
      "index.js",
      "kernel.spv",
      "kernel.wgsl.gz",
      "package.json",
    ]);
    assert.deepEqual(await names(join(root, "build", "diagnostics")), [
      "flow.ir",
      "kernel.ir",
      "kernel.spvasm",
      "project.json",
      "runtime.json",
      "spirv.kernel.ir",
      "spirv.plan.json",
      "web.kernel.ir",
      "web.plan.json",
    ]);
    for (const [name, contents] of ordinary) {
      assert.deepEqual(
        await Deno.readFile(join(root, "build", name)),
        contents,
      );
    }
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("check writes nothing and a failed build preserves complete output", async () => {
  const root = await fixture(source);
  try {
    await build({ cwd: root });
    const before = await Deno.readFile(join(root, "build", "index.js")),
      inventory = await names(join(root, "build"));
    await check({ cwd: root });
    assert.deepEqual(await names(join(root, "build")), inventory);
    assert.deepEqual(
      await Deno.readFile(join(root, "build", "index.js")),
      before,
    );
    await Deno.writeTextFile(
      join(root, "kernels", "scale.tach"),
      "export function broken[",
    );
    await assert.rejects(build({ cwd: root }));
    assert.deepEqual(await names(join(root, "build")), inventory);
    assert.deepEqual(
      await Deno.readFile(join(root, "build", "index.js")),
      before,
    );
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("docs are atomic and formatting is project-wide", async () => {
  const root = await fixture(source);
  try {
    await build({ cwd: root });
    const artifacts = [
      "index.d.ts",
      "index.js",
      "kernel.spv",
      "kernel.wgsl.gz",
      "package.json",
    ];
    const preserved = new Map(
      await Promise.all(
        artifacts.map(async (
          name,
        ) => [name, await Deno.readFile(join(root, "build", name))]),
      ),
    );
    await Deno.writeTextFile(
      join(root, "build", "docs", "stale.md"),
      "stale\n",
    );
    await Deno.writeTextFile(
      join(root, "tach.json"),
      `${
        JSON.stringify(
          {
            ...manifest,
            docs: { ...manifest.docs, summary: "Updated summary." },
          },
          null,
          2,
        )
      }\n`,
    );
    await docs({ cwd: root });
    assert.match(
      await Deno.readTextFile(join(root, "build", "README.md")),
      /Updated summary\./u,
    );
    assert.deepEqual(await names(join(root, "build", "docs")), ["kernels.md"]);
    for (const [name, contents] of preserved) {
      assert.deepEqual(
        await Deno.readFile(join(root, "build", name)),
        contents,
      );
    }
    const kernel = join(root, "kernels", "scale.tach"),
      readme = await Deno.readFile(join(root, "build", "README.md"));
    await Deno.writeTextFile(
      kernel,
      `@docs(title("Missing summary"));\n${source}`,
    );
    await assert.rejects(docs({ cwd: root }));
    assert.deepEqual(
      await Deno.readFile(join(root, "build", "README.md")),
      readme,
    );
    await Deno.writeTextFile(
      kernel,
      `// preserved\n${source.replaceAll("  ", "    ")}`,
    );
    await format({ cwd: join(root, "kernels") });
    const once = await Deno.readTextFile(kernel);
    assert.match(once, /\/\/ preserved/u);
    await format({ cwd: root });
    assert.equal(await Deno.readTextFile(kernel), once);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("docs creates only documentation before the first build", async () => {
  const root = await fixture(source);
  try {
    await docs({ cwd: root });
    assert.deepEqual(await names(join(root, "build")), ["README.md", "docs"]);
    assert.deepEqual(await names(join(root, "build", "docs")), ["kernels.md"]);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("generated documentation usage checks against the package facade", async () => {
  const root = await fixture(source, join(packageRoot, "tests"));
  try {
    await Deno.mkdir(join(root, "data"));
    await Deno.writeTextFile(
      join(root, "data", "extra.tach"),
      `@docs(title("Shared data"), summary("Additional documented types."));\n@docs(summary("An extra host value."), field(value, "Magnitude | normalized."))\ntype Extra = { value: float32 };\n`,
    );
    await build({ cwd: root });
    const readme = await Deno.readTextFile(join(root, "build", "README.md"));
    assert.match(readme, /^# Fixture kernels/mu);
    assert.ok(
      readme.indexOf("docs/data.md") < readme.indexOf("docs/kernels.md"),
    );
    assert.deepEqual(await names(join(root, "build", "docs")), [
      "data.md",
      "kernels.md",
    ]);
    assert.match(
      await Deno.readTextFile(join(root, "build", "docs", "data.md")),
      /## Shared data[\s\S]*Magnitude \\\| normalized\./u,
    );
    const snippet = readme.match(/```ts\n([\s\S]*?)```/u)?.[1];
    assert.ok(snippet);
    await Deno.writeTextFile(join(root, "build", "usage.ts"), snippet);
    const checked = command(Deno.execPath(), [
      "check",
      "--node-modules-dir=manual",
      "usage.ts",
    ], join(root, "build"));
    assert.equal(checked.status, 0, checked.stdout + checked.stderr);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("TACH_BIN must name an executable file", async () => {
  const directory = await Deno.makeTempDir({ prefix: "tach-bin-test-" }),
    previous = Deno.env.get("TACH_BIN");
  Deno.env.set("TACH_BIN", directory);
  try {
    await assert.rejects(compilerPath(), (error) => {
      assert.equal(error.code, "compiler-install");
      return true;
    });
  } finally {
    if (previous === undefined) Deno.env.delete("TACH_BIN");
    else Deno.env.set("TACH_BIN", previous);
    await Deno.remove(directory, { recursive: true });
  }
});

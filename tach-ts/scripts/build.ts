const root = new URL("../", import.meta.url), output = new URL("dist/", root);

interface GraphModule {
  readonly specifier: string;
  readonly dependencies: readonly {
    readonly code?: { readonly specifier: string };
    readonly type?: { readonly specifier: string };
  }[];
}

for (
  const entry of [
    "src/index.ts",
    "src/internal.ts",
    "src/compiler.ts",
    "declarations/index.ts",
    "declarations/compiler.ts",
    "src/cli.ts",
    "scripts/instructions.ts",
  ]
) {
  const result = await new Deno.Command(Deno.execPath(), {
    args: ["info", "--json", entry],
    cwd: root,
  }).output();
  if (!result.success) {
    throw new Error(`deno info ${entry} exited ${result.code}`);
  }
  const modules = new Map(
    (JSON.parse(new TextDecoder().decode(result.stdout))
      .modules as GraphModule[])
      .filter((module) => module.specifier.startsWith(root.href)).map((
        module,
      ) => [module.specifier, module]),
  );
  const visiting = new Set<string>(), visited = new Set<string>();
  const visit = (specifier: string, path: readonly string[]): void => {
    if (visiting.has(specifier)) {
      throw new Error(
        `cyclic module imports: ${
          [...path, specifier].map((value) => value.slice(root.href.length))
            .join(" -> ")
        }`,
      );
    }
    if (visited.has(specifier)) return;
    visiting.add(specifier);
    for (const dependency of modules.get(specifier)?.dependencies ?? []) {
      for (
        const resolved of [
          dependency.code?.specifier,
          dependency.type?.specifier,
        ]
      ) {
        if (resolved && modules.has(resolved)) {
          visit(resolved, [...path, specifier]);
        }
      }
    }
    visiting.delete(specifier);
    visited.add(specifier);
  };
  for (const specifier of modules.keys()) visit(specifier, []);
}

await Deno.remove(output, { recursive: true }).catch((error) => {
  if (!(error instanceof Deno.errors.NotFound)) throw error;
});
const result = await new Deno.Command(Deno.execPath(), {
  args: [
    "bundle",
    "--quiet",
    "--platform",
    "deno",
    "--code-splitting",
    "--outdir",
    "dist",
    "src/index.ts",
    "src/internal.ts",
    "src/compiler.ts",
  ],
  cwd: root,
  stdin: "null",
  stdout: "inherit",
  stderr: "inherit",
}).output();
if (!result.success) throw new Error(`deno bundle exited ${result.code}`);
const cli = await new Deno.Command(Deno.execPath(), {
  args: [
    "bundle",
    "--quiet",
    "--platform",
    "deno",
    "--outdir",
    "dist",
    "src/cli.ts",
  ],
  cwd: root,
  stdin: "null",
  stdout: "inherit",
  stderr: "inherit",
}).output();
if (!cli.success) throw new Error(`CLI bundle exited ${cli.code}`);
const declarations = await new Deno.Command(Deno.execPath(), {
  args: [
    "bundle",
    "--quiet",
    "--platform",
    "deno",
    "--declaration",
    "--outdir",
    "dist/declarations",
    "declarations/index.ts",
    "declarations/compiler.ts",
  ],
  cwd: root,
  stdin: "null",
  stdout: "inherit",
  stderr: "inherit",
}).output();
if (!declarations.success) {
  throw new Error(`declaration build exited ${declarations.code}`);
}
for (const name of ["index", "compiler"]) {
  await Deno.copyFile(
    new URL(`dist/declarations/${name}.d.ts`, root),
    new URL(`dist/${name}.d.ts`, root),
  );
}
await Deno.remove(new URL("dist/declarations/", root), { recursive: true });

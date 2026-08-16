import { normalizeError, TachError } from "./api.ts";
import {
  type Documentation,
  type DocumentedType,
  type FunctionDoc,
  type Parameter,
  renderDocumentation,
  typeScriptType,
} from "./docs.ts";

export interface CompilerRunOptions {
  readonly cwd?: string;
  readonly env?: Readonly<Record<string, string>>;
}

export interface BuildOptions extends CompilerRunOptions {
  readonly verbose?: boolean;
}

interface RuntimeProgram {
  readonly name: string;
  readonly parameters: readonly unknown[];
}

interface RuntimeTarget {
  readonly kernels: readonly unknown[];
  readonly programs: readonly unknown[];
  readonly vulkan?: string;
  readonly spirv?: string;
  readonly features?: readonly string[];
}

interface RuntimeMetadata {
  readonly schema: 2;
  readonly types: readonly unknown[];
  readonly programs: readonly RuntimeProgram[];
  readonly targets: {
    readonly web: RuntimeTarget;
    readonly spirv: RuntimeTarget;
  };
}

interface CompilerRun {
  readonly path: string;
  readonly stdout: string;
  readonly stderr: string;
}

interface NativeTarget {
  readonly executable: "tach" | "tach.exe";
  readonly asset: string;
}

const decoder = new TextDecoder();
const packageRoot = fromFileURL(new URL("../", import.meta.url));
const repositoryRoot = parent(packageRoot);
const nativeDirectory = join(packageRoot, "native");

async function sha256(bytes: Uint8Array): Promise<string> {
  return Array.from(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", bytes.slice().buffer),
    ),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function fromFileURL(url: URL): string {
  const path = decodeURIComponent(url.pathname);
  return Deno.build.os === "windows"
    ? path.replace(/^\/(.:)/u, "$1").replaceAll("/", "\\")
    : path;
}

function join(root: string, ...parts: readonly string[]): string {
  return [
    root.replace(/[\\/]+$/u, ""),
    ...parts.map((part) => part.replace(/^[\\/]+|[\\/]+$/gu, "")),
  ].filter(Boolean).join("/");
}

function parent(path: string): string {
  const normalized = path.replaceAll("\\", "/").replace(/\/$/u, ""),
    index = normalized.lastIndexOf("/");
  if (index === 0) return "/";
  if (index === 2 && /^[A-Za-z]:/u.test(normalized)) {
    return normalized.slice(0, 3);
  }
  return index < 0 ? normalized : normalized.slice(0, index);
}

function relative(root: string, path: string): string {
  const base = root.replaceAll("\\", "/").replace(/\/$/u, ""),
    value = path.replaceAll("\\", "/");
  if (!value.startsWith(`${base}/`)) {
    throw new Error(`${path} is outside ${root}`);
  }
  return value.slice(base.length + 1);
}

function absolute(path: string): string {
  return /^(?:[A-Za-z]:[\\/]|\/)/u.test(path) ? path : join(Deno.cwd(), path);
}

function missing(error: unknown): boolean {
  return error instanceof Deno.errors.NotFound;
}

async function remove(path: string): Promise<void> {
  try {
    await Deno.remove(path, { recursive: true });
  } catch (error) {
    if (!missing(error)) throw error;
  }
}

async function copyDirectory(
  source: string,
  destination: string,
): Promise<void> {
  await Deno.mkdir(destination);
  for await (const entry of Deno.readDir(source)) {
    const from = join(source, entry.name), to = join(destination, entry.name);
    if (entry.isDirectory) await copyDirectory(from, to);
    else if (entry.isFile) await Deno.copyFile(from, to);
    else throw new Error(`refusing to copy non-file ${from}`);
  }
}

async function readableExecutable(path: string): Promise<boolean> {
  try {
    const info = await Deno.stat(path);
    return info.isFile &&
      (Deno.build.os === "windows" || info.mode === null ||
        (info.mode & 0o111) !== 0);
  } catch {
    return false;
  }
}

function target(): NativeTarget {
  const targets: Readonly<Record<string, NativeTarget>> = {
    "windows:x86_64": {
      executable: "tach.exe",
      asset: "tach-windows-amd64.exe",
    },
    "windows:aarch64": {
      executable: "tach.exe",
      asset: "tach-windows-arm64.exe",
    },
    "linux:x86_64": { executable: "tach", asset: "tach-linux-amd64" },
    "linux:aarch64": { executable: "tach", asset: "tach-linux-arm64" },
    "darwin:aarch64": { executable: "tach", asset: "tach-darwin-arm64" },
  };
  const native = targets[Deno.build.os + ":" + Deno.build.arch];
  if (!native) {
    throw new TachError(
      "compiler-platform",
      `Tach does not publish a compiler for ${Deno.build.os}/${Deno.build.arch}`,
      { operation: "compilerPath" },
    );
  }
  return native;
}

export async function packageVersion(): Promise<string> {
  const info = JSON.parse(
    await Deno.readTextFile(join(packageRoot, "package.json")),
  ) as unknown;
  const version = typeof info === "object" && info !== null && "version" in info
    ? (info as { readonly version?: unknown }).version
    : undefined;
  if (typeof version !== "string" || version.length === 0) {
    throw new TypeError("@depths/tach package.json has no version");
  }
  return version;
}

async function developmentCompiler(
  native: NativeTarget,
): Promise<string | undefined> {
  const goModule = join(repositoryRoot, "go.mod");
  const candidate = join(repositoryRoot, "dist", native.executable);
  try {
    if (!(await Deno.stat(goModule)).isFile) return undefined;
  } catch {
    return undefined;
  }
  return await readableExecutable(candidate) ? candidate : undefined;
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  let failure: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(url, {
        redirect: "follow",
        headers: { "user-agent": "@depths/tach compiler installer" },
        signal: AbortSignal.timeout(30_000),
      });
      if (!response.ok) {
        throw new Error(`${url} returned HTTP ${response.status}`);
      }
      return new Uint8Array(await response.arrayBuffer());
    } catch (cause) {
      failure = cause;
      if (attempt < 2) {
        await new Promise((resolveDelay) =>
          setTimeout(resolveDelay, 250 * 2 ** attempt)
        );
      }
    }
  }
  throw failure;
}

function expectedHash(checksums: string, asset: string): string | undefined {
  for (const line of checksums.split(/\r?\n/u)) {
    const fields = line.trim().split(/\s+/u);
    if (fields.length >= 2 && fields[1]?.replace(/^[*]/u, "") === asset) {
      return fields[0]?.toLowerCase();
    }
  }
  return undefined;
}

async function installCompiler(
  native: NativeTarget,
  version: string,
): Promise<string> {
  const repository = Deno.env.get("TACH_GITHUB_REPOSITORY") ?? "Depths-AI/tach";
  const releaseBase =
    `https://github.com/${repository}/releases/download/v${version}`;

  try {
    const [binary, checksumBytes] = await Promise.all([
      fetchBytes(`${releaseBase}/${native.asset}`),
      fetchBytes(`${releaseBase}/checksums.txt`),
    ]);
    const checksums = new TextDecoder().decode(checksumBytes);
    const expected = expectedHash(checksums, native.asset);
    if (!expected) {
      throw new Error(`${native.asset} is missing from checksums.txt`);
    }
    const actual = await sha256(binary);
    if (actual !== expected) {
      throw new Error(
        `checksum mismatch for ${native.asset}: expected ${expected}, received ${actual}`,
      );
    }

    await Deno.mkdir(nativeDirectory, { recursive: true });
    const destination = join(nativeDirectory, native.executable);
    const temporary = join(
      nativeDirectory,
      `.${native.executable}.${crypto.randomUUID()}.tmp`,
    );
    try {
      await Deno.writeFile(temporary, binary, { mode: 0o755, createNew: true });
      if (Deno.build.os !== "windows") await Deno.chmod(temporary, 0o755);
      try {
        await Deno.rename(temporary, destination);
      } catch (cause) {
        // A concurrent npm install may have won the same atomic placement.
        if (!await readableExecutable(destination)) throw cause;
      }
    } finally {
      await remove(temporary);
    }
    return destination;
  } catch (cause) {
    throw normalizeError(cause, "compiler-install", "compilerPath");
  }
}

export async function compilerPath(): Promise<string> {
  let path: string;
  const override = Deno.env.get("TACH_BIN");
  if (override) {
    path = absolute(override);
    if (await readableExecutable(path)) {
      await verifyCompilerVersion(path);
      return path;
    }
    throw new TachError(
      "compiler-install",
      `TACH_BIN does not point to an executable: ${path}`,
      { operation: "compilerPath" },
    );
  }

  const native = target();

  const installed = join(nativeDirectory, native.executable);
  if (await readableExecutable(installed)) path = installed;
  else {
    const development = await developmentCompiler(native);
    if (development) path = development;
    else {
      try {
        path = await installCompiler(native, await packageVersion());
      } catch (cause) {
        throw normalizeError(cause, "compiler-install", "compilerPath");
      }
    }
  }
  await verifyCompilerVersion(path);
  return path;
}

async function verifyCompilerVersion(path: string): Promise<void> {
  const expected = await packageVersion();
  const result = await new Deno.Command(path, { args: ["_version"] }).output();
  if (!result.success) {
    throw new Error(
      decoder.decode(result.stderr).trim() || `compiler exited ${result.code}`,
    );
  }
  const version = decoder.decode(result.stdout).trim();
  if (version !== expected) {
    throw new TachError(
      "compiler-install",
      `compiler version ${version} does not match @depths/tach ${expected}`,
      { operation: "compilerPath" },
    );
  }
}

async function runNative(
  args: readonly string[],
  options: CompilerRunOptions = {},
): Promise<CompilerRun> {
  const path = await compilerPath();
  try {
    const command = new Deno.Command(path, {
      args: [...args],
      ...(options.cwd ? { cwd: options.cwd } : {}),
      ...(options.env ? { env: options.env } : {}),
    });
    const result = await command.output();
    const output = {
      path,
      stdout: decoder.decode(result.stdout),
      stderr: decoder.decode(result.stderr),
    };
    if (result.success) return output;
    throw new TachError(
      "compiler-execution",
      `${args[0]} failed with exit code ${result.code}${
        output.stderr ? `\n${output.stderr.trimEnd()}` : ""
      }`,
      { operation: "compiler", cause: { ...output, exitCode: result.code } },
    );
  } catch (cause) {
    if (cause instanceof TachError) throw cause;
    throw normalizeError(cause, "compiler-execution", "compiler");
  }
}

async function projectRoot(cwd = Deno.cwd()): Promise<string> {
  let directory = await Deno.realPath(cwd);
  for (;;) {
    try {
      if ((await Deno.stat(join(directory, "tach.json"))).isFile) {
        return directory;
      }
    } catch { /* Keep walking until the nearest project manifest is found. */ }
    const ancestor = parent(directory);
    if (ancestor === directory) {
      throw new TachError(
        "compiler-execution",
        `no tach.json found from ${cwd}`,
        { operation: "project" },
      );
    }
    directory = ancestor;
  }
}

function parseDescription(stdout: string): Documentation {
  try {
    return JSON.parse(stdout) as Documentation;
  } catch (cause) {
    throw normalizeError(cause, "compiler-execution", "description");
  }
}

function parseRuntime(source: string): RuntimeMetadata {
  const runtime = JSON.parse(source) as RuntimeMetadata;
  if (
    runtime.schema !== 2 || !Array.isArray(runtime.programs) ||
    !runtime.targets?.web || !runtime.targets.spirv
  ) {
    throw new TypeError("invalid Tach runtime metadata");
  }
  const spirv = runtime.targets.spirv;
  if (
    spirv.vulkan !== "1.3" || spirv.spirv !== "1.6" ||
    spirv.features?.join("\n") !==
      "synchronization2\nshaderZeroInitializeWorkgroupMemory"
  ) {
    throw new TypeError("invalid Tach Vulkan 1.3/SPIR-V 1.6 profile");
  }
  return runtime;
}

function generatedPackage(
  project: Documentation,
  runtimeVersion: string,
): object {
  return {
    name: project.package,
    version: project.version,
    type: "module",
    sideEffects: false,
    exports: {
      ".": {
        types: "./index.d.ts",
        default: "./index.js",
      },
    },
    dependencies: { "@depths/tach": runtimeVersion },
  };
}

function documentedPrograms(
  project: Documentation,
  runtime: RuntimeMetadata,
): readonly FunctionDoc[] {
  const functions = new Map(
    project.modules.flatMap((module) =>
      module.kernels.flatMap((kernel) =>
        kernel.functions.map((fn) => [fn.name, fn] as const)
      )
    ),
  );
  return runtime.programs.map((program) => {
    const fn = functions.get(program.name);
    if (!fn?.exported || fn.parameters.length !== program.parameters.length) {
      throw new TypeError(
        `runtime program ${
          JSON.stringify(program.name)
        } does not match project documentation`,
      );
    }
    return fn;
  });
}

function generatedModule(runtime: RuntimeMetadata, webVersion: string): string {
  const definition = JSON.stringify({
    schema: runtime.schema,
    types: runtime.types,
    programs: runtime.programs,
    targets: runtime.targets,
  });
  let source =
    `// Generated by Tach.\nimport { defineModule as $defineModule } from "@depths/tach/internal";\n\nconst $tach = $defineModule({ ...${definition}, shaders: { web: new URL("./kernel.wgsl.gz?v=${webVersion}", import.meta.url), spirv: new URL("./kernel.spv", import.meta.url) } });\n\n`;
  runtime.programs.forEach((program, index) => {
    source +=
      `export function ${program.name}(...$args) { const $options = $args.length > ${program.parameters.length} ? $args.pop() : undefined; return $tach.command(${index}, $args, $options); }\n`;
  });
  return source;
}

function jsDoc(
  indent: string,
  summary: string | undefined,
  tags: readonly string[] = [],
): string {
  if (!summary && tags.length === 0) return "";
  let out = `${indent}/**\n`;
  for (const entry of [summary ?? "", ...tags]) {
    for (const line of entry.split("\n")) {
      out += `${indent} * ${line.replaceAll("*/", "*\\/")}\n`;
    }
  }
  return `${out}${indent} */\n`;
}

function bufferContext(parameter: Parameter): string {
  return parameter.access === "atomic"
    ? "Atomic GPU buffer."
    : parameter.access === "readWrite"
    ? "Read/write GPU buffer."
    : parameter.access === "write"
    ? "Output GPU buffer."
    : "Read-only GPU buffer.";
}

function generatedType(type: DocumentedType): string {
  let out = jsDoc("", type.summary) + `export type ${type.name} = {\n`;
  for (const field of type.fields) {
    out += jsDoc("  ", field.description);
    const name = /^[A-Za-z_$][A-Za-z0-9_$]*$/u.test(field.name)
      ? field.name
      : JSON.stringify(field.name);
    out += `  readonly ${name}: ${typeScriptType(field.type)};\n`;
  }
  return `${out}};\n\n`;
}

function generatedFunction(fn: FunctionDoc): string {
  const tags = [
    ...fn.coordinates.flatMap((coordinate) =>
      coordinate.description
        ? [
          `@remarks Coordinate \`${coordinate.name}\`: ${coordinate.description}`,
        ]
        : []
    ),
    ...fn.parameters.flatMap((parameter) => {
      const description = `${
        parameter.buffer ? `${bufferContext(parameter)} ` : ""
      }${parameter.description ?? ""}`.trim();
      return description ? [`@param ${parameter.name} ${description}`] : [];
    }),
  ];
  let out = jsDoc("", fn.summary, tags) + `export function ${fn.name}(\n`;
  for (const parameter of fn.parameters) {
    const type = typeScriptType(parameter.type);
    out += `  ${parameter.name}: ${
      parameter.buffer ? `$ComputeBuffer<${type}>` : type
    },\n`;
  }
  if (fn.coordinates.length === 0) out += "  $options?: $CommandOptions,\n";
  else {
    const size = fn.coordinates.length === 1
      ? "number"
      : `readonly [${
        fn.coordinates.map((coordinate) => `${coordinate.name}: number`).join(
          ", ",
        )
      }]`;
    out += `  $launch?: $LaunchOptions<${size}>,\n`;
  }
  return `${out}): ${
    fn.returns?.type.kind === "view" ? "$ComputeView" : "$ComputeCommand"
  };\n\n`;
}

function generatedDeclarations(
  project: Documentation,
  runtime: RuntimeMetadata,
): string {
  let out =
    '// Generated by Tach. Typed host-independent GPU module.\n\nimport type { CommandOptions as $CommandOptions, ComputeBuffer as $ComputeBuffer, ComputeCommand as $ComputeCommand, ComputeView as $ComputeView, LaunchOptions as $LaunchOptions } from "@depths/tach";\n\n';
  for (
    const type of project.modules.flatMap((module) =>
      module.kernels.flatMap((kernel) => kernel.types)
    )
  ) out += generatedType(type);
  for (const fn of documentedPrograms(project, runtime)) {
    out += generatedFunction(fn);
  }
  return out;
}

async function writeDocumentation(
  root: string,
  project: Documentation,
): Promise<void> {
  const rendered = renderDocumentation(project);
  const directory = join(root, "docs");
  await Deno.mkdir(directory, { recursive: true });
  await Deno.writeTextFile(join(root, "README.md"), rendered.readme);
  await Promise.all(
    [...rendered.modules].map(([name, markdown]) =>
      Deno.writeTextFile(join(directory, `${name}.md`), markdown)
    ),
  );
}

async function inventory(root: string): Promise<string[]> {
  const files: string[] = [];
  async function visit(directory: string): Promise<void> {
    for await (const entry of Deno.readDir(directory)) {
      const path = join(directory, entry.name);
      if (entry.isDirectory) await visit(path);
      else if (entry.isFile) files.push(relative(root, path));
      else throw new Error(`invalid staged artifact ${path}`);
    }
  }
  await visit(root);
  return files.sort();
}

function expectedInventory(project: Documentation, verbose: boolean): string[] {
  const files = [
    "README.md",
    ...project.modules.map((module) => `docs/${module.name}.md`),
    "index.d.ts",
    "index.js",
    "kernel.spv",
    "kernel.wgsl.gz",
    "package.json",
  ];
  if (verbose) {
    files.push(
      ...[
        "flow.ir",
        "kernel.ir",
        "kernel.spvasm",
        "project.json",
        "runtime.json",
        "spirv.kernel.ir",
        "spirv.plan.json",
        "web.kernel.ir",
        "web.plan.json",
      ].map((name) => `diagnostics/${name}`),
    );
  }
  return files.sort();
}

async function assertInventory(
  stage: string,
  project: Documentation,
  verbose: boolean,
): Promise<void> {
  const actual = await inventory(stage);
  const expected = expectedInventory(project, verbose);
  if (actual.join("\n") !== expected.join("\n")) {
    throw new Error(`invalid staged artifact inventory:\n${actual.join("\n")}`);
  }
}

async function replaceBuild(root: string, stage: string): Promise<void> {
  const destination = join(root, "build");
  const backup = join(root, `.build.${crypto.randomUUID()}.backup`);
  let previous = false;
  try {
    const info = await Deno.lstat(destination);
    if (!info.isDirectory || info.isSymlink) {
      throw new Error(`${destination} is not a replaceable build directory`);
    }
    await Deno.rename(destination, backup);
    previous = true;
  } catch (cause) {
    if (!missing(cause)) throw cause;
  }
  try {
    await Deno.rename(stage, destination);
  } catch (cause) {
    if (previous) await Deno.rename(backup, destination);
    throw cause;
  }
  if (previous) await remove(backup);
}

export interface ProjectResult {
  readonly root: string;
  readonly description: Documentation;
}

export async function build(
  options: BuildOptions = {},
): Promise<ProjectResult> {
  const { verbose = false, ...runOptions } = options;
  const root = await projectRoot(runOptions.cwd);
  const stage = join(root, `.build.${crypto.randomUUID()}.tmp`);
  await Deno.mkdir(stage);
  try {
    const args = ["_build", "--output", stage];
    if (verbose) args.push("--verbose");
    await runNative(args, { ...runOptions, cwd: root });
    const [projectSource, runtimeSource] = await Promise.all([
      Deno.readTextFile(join(stage, "project.json")),
      Deno.readTextFile(join(stage, "runtime.json")),
    ]);
    const description = parseDescription(projectSource);
    const runtime = parseRuntime(runtimeSource);
    const webVersion = await sha256(
      await Deno.readFile(join(stage, "kernel.wgsl.gz")),
    );
    await writeDocumentation(stage, description);
    await Promise.all([
      Deno.writeTextFile(
        join(stage, "index.js"),
        generatedModule(runtime, webVersion),
      ),
      Deno.writeTextFile(
        join(stage, "index.d.ts"),
        generatedDeclarations(description, runtime),
      ),
      Deno.writeTextFile(
        join(stage, "package.json"),
        `${
          JSON.stringify(
            generatedPackage(description, await packageVersion()),
            null,
            2,
          )
        }\n`,
      ),
    ]);
    if (verbose) {
      await Deno.rename(
        join(stage, "project.json"),
        join(stage, "diagnostics", "project.json"),
      );
      await Deno.rename(
        join(stage, "runtime.json"),
        join(stage, "diagnostics", "runtime.json"),
      );
    } else {
      await Promise.all([
        remove(join(stage, "project.json")),
        remove(join(stage, "runtime.json")),
      ]);
    }
    await assertInventory(stage, description, verbose);
    await replaceBuild(root, stage);
    return { root, description };
  } catch (cause) {
    await remove(stage);
    throw cause;
  }
}

export async function check(
  options: CompilerRunOptions = {},
): Promise<ProjectResult> {
  const root = await projectRoot(options.cwd);
  const run = await runNative(["_check"], { ...options, cwd: root });
  const result = JSON.parse(run.stdout) as {
    readonly project: Documentation;
    readonly runtime: RuntimeMetadata;
  };
  const description = parseDescription(JSON.stringify(result.project));
  const runtime = parseRuntime(JSON.stringify(result.runtime));
  renderDocumentation(description);
  generatedDeclarations(description, runtime);
  generatedModule(runtime, "check");
  JSON.stringify(generatedPackage(description, await packageVersion()));
  return { root, description };
}

export async function docs(
  options: CompilerRunOptions = {},
): Promise<ProjectResult> {
  const root = await projectRoot(options.cwd);
  const run = await runNative(["_docs"], { ...options, cwd: root });
  const description = parseDescription(run.stdout);
  const stage = join(root, `.build.${crypto.randomUUID()}.tmp`);
  try {
    try {
      const current = join(root, "build");
      const info = await Deno.lstat(current);
      if (!info.isDirectory || info.isSymlink) {
        throw new Error(`${current} is not a build directory`);
      }
      await copyDirectory(current, stage);
    } catch (cause) {
      if (!missing(cause)) throw cause;
      await Deno.mkdir(stage);
    }
    await remove(join(stage, "README.md"));
    await remove(join(stage, "docs"));
    await writeDocumentation(stage, description);
    await replaceBuild(root, stage);
    return { root, description };
  } catch (cause) {
    await remove(stage);
    throw cause;
  }
}

export async function format(options: CompilerRunOptions = {}): Promise<void> {
  const root = await projectRoot(options.cwd);
  await runNative(["_fmt"], { ...options, cwd: root });
}

import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, chmod, cp, lstat, mkdir, readFile, readdir, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

import { normalizeError, TachError } from "./error.js";
import { renderDocumentation, type Documentation } from "./docs.js";

export interface CompilerRunOptions {
  readonly cwd?: string;
  readonly env?: Readonly<Record<string, string>>;
}

export type BuildTarget = "web" | "spirv";

export interface BuildOptions extends CompilerRunOptions {
  readonly target?: BuildTarget;
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

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(packageRoot, "..");
const nativeDirectory = join(packageRoot, "native");

async function readableExecutable(path: string): Promise<boolean> {
  try {
    if (!(await stat(path)).isFile()) return false;
    await access(path, process.platform === "win32" ? constants.F_OK : constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function target(): NativeTarget {
  const os = process.platform === "win32"
    ? "windows"
    : process.platform === "darwin" || process.platform === "linux"
      ? process.platform
      : undefined;
  const arch = process.arch === "x64"
    ? "amd64"
    : process.arch === "arm64"
      ? "arm64"
      : undefined;
  if (!os || !arch) {
    throw new TachError(
      "compiler-platform",
      `Tach does not publish a compiler for ${process.platform}/${process.arch}`,
      { operation: "compilerPath" },
    );
  }
  const executable = os === "windows" ? "tach.exe" : "tach";
  return {
    executable,
    asset: `tach-${os}-${arch}${os === "windows" ? ".exe" : ""}`,
  };
}

export async function packageVersion(): Promise<string> {
  const info = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8")) as unknown;
  const version = typeof info === "object" && info !== null && "version" in info
    ? (info as { readonly version?: unknown }).version
    : undefined;
  if (typeof version !== "string" || version.length === 0) {
    throw new TypeError("@depths/tach package.json has no version");
  }
  return version;
}

async function developmentCompiler(native: NativeTarget): Promise<string | undefined> {
  const goModule = join(repositoryRoot, "go.mod");
  const candidate = join(repositoryRoot, "dist", native.executable);
  try {
    await access(goModule, constants.R_OK);
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
      if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    } catch (cause) {
      failure = cause;
      if (attempt < 2) {
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 250 * 2 ** attempt));
      }
    }
  }
  throw failure;
}

function expectedHash(checksums: string, asset: string): string | undefined {
  for (const line of checksums.split(/\r?\n/u)) {
    const fields = line.trim().split(/\s+/u);
    if (fields.length >= 2 && fields[1]?.replace(/^[*]/u, "") === asset) return fields[0]?.toLowerCase();
  }
  return undefined;
}

async function installCompiler(native: NativeTarget, version: string): Promise<string> {
  if (version === "0.0.0") {
    throw new TachError(
      "compiler-install",
      "development compiler is missing; run `npm run compiler` at the Tach repository root",
      { operation: "compilerPath" },
    );
  }
  const repository = process.env.TACH_GITHUB_REPOSITORY ?? "Depths-AI/tach";
  const releaseBase = `https://github.com/${repository}/releases/download/v${version}`;

  try {
    const [binary, checksumBytes] = await Promise.all([
      fetchBytes(`${releaseBase}/${native.asset}`),
      fetchBytes(`${releaseBase}/checksums.txt`),
    ]);
    const checksums = new TextDecoder().decode(checksumBytes);
    const expected = expectedHash(checksums, native.asset);
    if (!expected) throw new Error(`${native.asset} is missing from checksums.txt`);
    const actual = createHash("sha256").update(binary).digest("hex");
    if (actual !== expected) {
      throw new Error(`checksum mismatch for ${native.asset}: expected ${expected}, received ${actual}`);
    }

    await mkdir(nativeDirectory, { recursive: true });
    const destination = join(nativeDirectory, native.executable);
    const temporary = join(nativeDirectory, `.${native.executable}.${randomUUID()}.tmp`);
    try {
      await writeFile(temporary, binary, { mode: 0o755, flag: "wx" });
      if (process.platform !== "win32") await chmod(temporary, 0o755);
      try {
        await rename(temporary, destination);
      } catch (cause) {
        // A concurrent npm install may have won the same atomic placement.
        if (!await readableExecutable(destination)) throw cause;
      }
    } finally {
      await rm(temporary, { force: true });
    }
    return destination;
  } catch (cause) {
    throw normalizeError(cause, "compiler-install", "compilerPath");
  }
}

export async function compilerPath(): Promise<string> {
  let path: string;
  const override = process.env.TACH_BIN;
  if (override) {
    path = isAbsolute(override) ? override : resolve(process.cwd(), override);
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
  const result = spawn(path, ["_version"], { stdio: ["ignore", "pipe", "pipe"] });
  const stdout: Uint8Array[] = [];
  const stderr: Uint8Array[] = [];
  result.stdout.on("data", (chunk: Uint8Array) => stdout.push(chunk));
  result.stderr.on("data", (chunk: Uint8Array) => stderr.push(chunk));
  const version = await new Promise<string>((resolveVersion, rejectVersion) => {
    result.once("error", rejectVersion);
    result.once("close", (code) => code === 0
      ? resolveVersion(Buffer.concat(stdout).toString("utf8").trim())
      : rejectVersion(new Error(Buffer.concat(stderr).toString("utf8").trim() || `compiler exited ${code}`)));
  });
  if (version !== expected && !(expected === "0.0.0" && version === "dev")) {
    throw new TachError("compiler-install", `compiler version ${version} does not match @depths/tach ${expected}`, { operation: "compilerPath" });
  }
}

async function runNative(args: readonly string[], options: CompilerRunOptions = {}): Promise<CompilerRun> {
  const path = await compilerPath();
  let child;
  try {
    child = spawn(path, [...args], { cwd: options.cwd, env: { ...process.env, ...options.env }, stdio: ["ignore", "pipe", "pipe"] });
  } catch (cause) {
    throw normalizeError(cause, "compiler-execution", "compiler");
  }
  return new Promise((resolveRun, rejectRun) => {
    const stdoutChunks: Uint8Array[] = [];
    const stderrChunks: Uint8Array[] = [];
    child.stdout.on("data", (chunk: Uint8Array) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk: Uint8Array) => stderrChunks.push(chunk));
    child.once("error", (cause) => rejectRun(normalizeError(cause, "compiler-execution", "compiler")));
    child.once("close", (code, signal) => {
      const output = { path, stdout: Buffer.concat(stdoutChunks).toString("utf8"), stderr: Buffer.concat(stderrChunks).toString("utf8") };
      if (code === 0) return resolveRun(output);
      const reason = signal ? `signal ${signal}` : `exit code ${code ?? "unknown"}`;
      rejectRun(new TachError("compiler-execution", `${args[0]} failed with ${reason}${output.stderr ? `\n${output.stderr.trimEnd()}` : ""}`, { operation: "compiler", cause: { ...output, exitCode: code, signal } }));
    });
  });
}

async function projectRoot(cwd = process.cwd()): Promise<string> {
  let directory = resolve(cwd);
  for (;;) {
    try {
      if ((await stat(join(directory, "tach.json"))).isFile()) return realpath(directory);
    } catch {}
    const parent = dirname(directory);
    if (parent === directory) throw new TachError("compiler-execution", `no tach.json found from ${cwd}`, { operation: "project" });
    directory = parent;
  }
}

function parseDescription(stdout: string): Documentation {
  try {
    return JSON.parse(stdout) as Documentation;
  } catch (cause) {
    throw normalizeError(cause, "compiler-execution", "description");
  }
}

function generatedPackage(project: Documentation, runtimeVersion: string): object {
  return {
    name: project.package,
    version: project.version,
    type: "module",
    sideEffects: false,
    exports: { ".": { types: "./index.d.ts", import: "./index.js" } },
    dependencies: { "@depths/tach": runtimeVersion },
  };
}

async function writeDocumentation(root: string, project: Documentation): Promise<void> {
  const rendered = renderDocumentation(project);
  const directory = join(root, "docs");
  await mkdir(directory, { recursive: true });
  await writeFile(join(root, "README.md"), rendered.readme, "utf8");
  await Promise.all([...rendered.modules].map(([name, markdown]) => writeFile(join(directory, `${name}.md`), markdown, "utf8")));
}

async function inventory(root: string): Promise<string[]> {
  const files: string[] = [];
  async function visit(directory: string): Promise<void> {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else files.push(relative(root, path).replaceAll("\\", "/"));
    }
  }
  await visit(root);
  return files.sort();
}

function expectedInventory(project: Documentation, target: BuildTarget): string[] {
  const files = ["README.md", ...project.modules.map((module) => `docs/${module.name}.md`)];
  files.push(...target === "web" ? ["index.d.ts", "index.js", "kernel.wgsl", "package.json"] : ["kernel.spv"]);
  return files.sort();
}

async function assertInventory(stage: string, project: Documentation, target: BuildTarget): Promise<void> {
  const actual = await inventory(stage);
  const expected = expectedInventory(project, target);
  if (actual.join("\n") !== expected.join("\n")) throw new Error(`invalid staged artifact inventory:\n${actual.join("\n")}`);
}

async function replaceBuild(root: string, stage: string): Promise<void> {
  const destination = join(root, "build");
  const backup = join(root, `.build.${randomUUID()}.backup`);
  let previous = false;
  try {
    const info = await lstat(destination);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`${destination} is not a replaceable build directory`);
    await rename(destination, backup);
    previous = true;
  } catch (cause) {
    if ((cause as NodeJS.ErrnoException).code !== "ENOENT") throw cause;
  }
  try {
    await rename(stage, destination);
  } catch (cause) {
    if (previous) await rename(backup, destination);
    throw cause;
  }
  if (previous) await rm(backup, { recursive: true, force: true });
}

export interface ProjectResult { readonly root: string; readonly description: Documentation }

export async function build(options: BuildOptions = {}): Promise<ProjectResult> {
  const { target = "web", ...runOptions } = options;
  const root = await projectRoot(runOptions.cwd);
  const stage = join(root, `.build.${randomUUID()}.tmp`);
  await mkdir(stage);
  try {
    const run = await runNative(["_build", "--target", target, "--output", stage], { ...runOptions, cwd: root });
    const description = parseDescription(run.stdout);
    await writeDocumentation(stage, description);
    if (target === "web") await writeFile(join(stage, "package.json"), `${JSON.stringify(generatedPackage(description, await packageVersion()), null, 2)}\n`, "utf8");
    await assertInventory(stage, description, target);
    await replaceBuild(root, stage);
    return { root, description };
  } catch (cause) {
    await rm(stage, { recursive: true, force: true });
    throw cause;
  }
}

export async function check(options: CompilerRunOptions = {}): Promise<ProjectResult> {
  const root = await projectRoot(options.cwd);
  const run = await runNative(["_check"], { ...options, cwd: root });
  const description = parseDescription(run.stdout);
  renderDocumentation(description);
  JSON.stringify(generatedPackage(description, await packageVersion()));
  return { root, description };
}

export async function docs(options: CompilerRunOptions = {}): Promise<ProjectResult> {
  const root = await projectRoot(options.cwd);
  const run = await runNative(["_docs"], { ...options, cwd: root });
  const description = parseDescription(run.stdout);
  const stage = join(root, `.build.${randomUUID()}.tmp`);
  try {
    try {
      const current = join(root, "build");
      const info = await lstat(current);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`${current} is not a build directory`);
      await cp(current, stage, { recursive: true, errorOnExist: true });
    } catch (cause) {
      if ((cause as NodeJS.ErrnoException).code !== "ENOENT") throw cause;
      await mkdir(stage);
    }
    await rm(join(stage, "README.md"), { force: true });
    await rm(join(stage, "docs"), { recursive: true, force: true });
    await writeDocumentation(stage, description);
    await replaceBuild(root, stage);
    return { root, description };
  } catch (cause) {
    await rm(stage, { recursive: true, force: true });
    throw cause;
  }
}

export async function format(options: CompilerRunOptions = {}): Promise<void> {
  const root = await projectRoot(options.cwd);
  await runNative(["_fmt"], { ...options, cwd: root });
}

import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

import { normalizeError, TachError } from "./error.js";
import { renderDocumentation, type Documentation } from "./docs.js";

export interface CompilerRunOptions {
  readonly cwd?: string;
  readonly env?: Readonly<Record<string, string>>;
}

export type BuildTarget = "web" | "spirv" | "all";

export interface BuildOptions extends CompilerRunOptions {
  readonly target?: BuildTarget;
}

export interface CompilerRun {
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

async function packageVersion(): Promise<string> {
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
  const override = process.env.TACH_BIN;
  if (override) {
    const path = isAbsolute(override) ? override : resolve(process.cwd(), override);
    if (await readableExecutable(path)) return path;
    throw new TachError(
      "compiler-install",
      `TACH_BIN does not point to an executable: ${path}`,
      { operation: "compilerPath" },
    );
  }

  const native = target();

  const installed = join(nativeDirectory, native.executable);
  if (await readableExecutable(installed)) return installed;
  const development = await developmentCompiler(native);
  if (development) return development;
  try {
    return await installCompiler(native, await packageVersion());
  } catch (cause) {
    throw normalizeError(cause, "compiler-install", "compilerPath");
  }
}

export async function runCompiler(
  args: readonly string[],
  options: CompilerRunOptions = {},
): Promise<CompilerRun> {
  const documentation = args[0] === "docs";
  if (documentation && args.length !== 2) {
    throw new TachError(
      "compiler-execution",
      "tach docs expects exactly one .tach file",
      { operation: "compiler" },
    );
  }
  const path = await compilerPath();

  let child;
  try {
    child = spawn(path, documentation ? ["describe", args[1]!] : [...args], {
      cwd: options.cwd,
      env: { ...process.env, ...options.env },
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (cause) {
    throw normalizeError(cause, "compiler-execution", "compiler");
  }
  return new Promise((resolveRun, rejectRun) => {
    const stdoutChunks: Uint8Array[] = [];
    const stderrChunks: Uint8Array[] = [];
    child.stdout.on("data", (chunk: Uint8Array) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk: Uint8Array) => stderrChunks.push(chunk));
    child.once("error", (cause) => {
      rejectRun(normalizeError(cause, "compiler-execution", "compiler"));
    });
    child.once("close", (code, signal) => {
      let stdout = Buffer.concat(stdoutChunks).toString("utf8");
      if (code === 0 && documentation) {
        try {
          stdout = renderDocumentation(JSON.parse(stdout) as Documentation);
        } catch (cause) {
          rejectRun(normalizeError(cause, "compiler-execution", "docs"));
          return;
        }
      }
      const output = {
        path,
        stdout,
        stderr: Buffer.concat(stderrChunks).toString("utf8"),
      };
      if (code === 0) {
        resolveRun(output);
        return;
      }
      const reason = signal ? `signal ${signal}` : `exit code ${code ?? "unknown"}`;
      rejectRun(new TachError(
        "compiler-execution",
        `tach ${args.join(" ")} failed with ${reason}`,
        {
          operation: "compiler",
          cause: { ...output, exitCode: code, signal },
        },
      ));
    });
  });
}

export async function build(
  source: string,
  options: BuildOptions = {},
): Promise<CompilerRun> {
  const { target, ...runOptions } = options;
  return runCompiler(target === undefined ? ["build", source] : ["build", "--target", target, source], runOptions);
}

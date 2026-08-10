import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

import { err, normalizeError, ok, tachError, type Result } from "./result.js";

export interface CompilerRunOptions {
  readonly cwd?: string;
  readonly env?: Readonly<Record<string, string>>;
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

function target(): Result<NativeTarget> {
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
    return err(tachError(
      "compiler-platform",
      `Tach does not publish a compiler for ${process.platform}/${process.arch}`,
      { operation: "compilerPath" },
    ));
  }
  const executable = os === "windows" ? "tach.exe" : "tach";
  return ok({
    executable,
    asset: `tach-${os}-${arch}${os === "windows" ? ".exe" : ""}`,
  });
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

async function installCompiler(native: NativeTarget, version: string): Promise<Result<string>> {
  if (version === "0.0.0") {
    return err(tachError(
      "compiler-install",
      "development compiler is missing; run `npm run compiler` at the Tach repository root",
      { operation: "compilerPath" },
    ));
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
    return ok(destination);
  } catch (cause) {
    return err(normalizeError(cause, "compiler-install", "compilerPath"));
  }
}

export async function compilerPath(): Promise<Result<string>> {
  const override = process.env.TACH_BIN;
  if (override) {
    const path = isAbsolute(override) ? override : resolve(process.cwd(), override);
    if (await readableExecutable(path)) return ok(path);
    return err(tachError(
      "compiler-install",
      `TACH_BIN does not point to an executable: ${path}`,
      { operation: "compilerPath" },
    ));
  }

  const selected = target();
  if (!selected.ok) return selected;
  const native = selected.value;

  const installed = join(nativeDirectory, native.executable);
  if (await readableExecutable(installed)) return ok(installed);
  const development = await developmentCompiler(native);
  if (development) return ok(development);
  try {
    return await installCompiler(native, await packageVersion());
  } catch (cause) {
    return err(normalizeError(cause, "compiler-install", "compilerPath"));
  }
}

export async function runCompiler(
  args: readonly string[],
  options: CompilerRunOptions = {},
): Promise<Result<CompilerRun>> {
  const resolved = await compilerPath();
  if (!resolved.ok) return resolved;
  const path = resolved.value;

  let child;
  try {
    child = spawn(path, [...args], {
      cwd: options.cwd,
      env: { ...process.env, ...options.env },
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (cause) {
    return err(normalizeError(cause, "compiler-execution", "compiler"));
  }
  return new Promise((resolveResult) => {
    const stdout: Uint8Array[] = [];
    const stderr: Uint8Array[] = [];
    child.stdout.on("data", (chunk: Uint8Array) => stdout.push(chunk));
    child.stderr.on("data", (chunk: Uint8Array) => stderr.push(chunk));
    child.once("error", (cause) => {
      resolveResult(err(normalizeError(cause, "compiler-execution", "compiler")));
    });
    child.once("close", (code, signal) => {
      const output = {
        path,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8"),
      };
      if (code === 0) {
        resolveResult(ok(output));
        return;
      }
      const reason = signal ? `signal ${signal}` : `exit code ${code ?? "unknown"}`;
      resolveResult(err(tachError(
        "compiler-execution",
        `tach ${args.join(" ")} failed with ${reason}`,
        {
          operation: "compiler",
          cause: { ...output, exitCode: code, signal },
        },
      )));
    });
  });
}

export async function build(
  source: string,
  options: CompilerRunOptions = {},
): Promise<Result<CompilerRun>> {
  return runCompiler(["build", source], options);
}

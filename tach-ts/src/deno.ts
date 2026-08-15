import type {
  Driver,
  DriverBuffer,
  ModuleDefinition,
  PreparedBarrierResource,
  PreparedCommand,
  PreparedResource,
} from "./driver.ts";
import { type TachAdapterInfo, TachError, type TachOptions } from "./api.ts";

type Pointer = object | null;
type NativeFunction = (
  ...arguments_: readonly unknown[]
) => number | Promise<number>;
interface DenoAPI {
  readonly build: { readonly os: string; readonly arch: string };
  readonly UnsafePointer: { create(value: bigint): Pointer };
  dlopen(
    path: URL,
    symbols: Readonly<Record<string, unknown>>,
  ): { readonly symbols: Readonly<Record<string, NativeFunction>> };
  readFile(path: URL): Promise<Uint8Array>;
}

const abi = 1, textDecoder = new TextDecoder(), textEncoder = new TextEncoder();
let nativeLibrary: ReturnType<DenoAPI["dlopen"]> | undefined;
const symbols = {
  tach_abi_version: { parameters: [], result: "u32" },
  tach_open: { parameters: ["buffer", "usize", "buffer"], result: "i32" },
  tach_info: { parameters: ["pointer", "buffer", "usize"], result: "i32" },
  tach_module: {
    parameters: ["pointer", "buffer", "usize", "buffer", "usize", "buffer"],
    result: "i32",
    nonblocking: true,
  },
  tach_buffer: {
    parameters: ["pointer", "u32", "buffer", "buffer"],
    result: "i32",
  },
  tach_write: {
    parameters: ["pointer", "u32", "buffer", "u32"],
    result: "i32",
  },
  tach_submit: {
    parameters: ["pointer", "buffer", "usize", "buffer"],
    result: "i32",
    nonblocking: true,
  },
  tach_wait: {
    parameters: ["pointer", "u64"],
    result: "i32",
    nonblocking: true,
  },
  tach_read: {
    parameters: ["pointer", "u32", "buffer", "u32"],
    result: "i32",
    nonblocking: true,
  },
  tach_destroy_buffer: { parameters: ["pointer", "u32"], result: "void" },
  tach_close: { parameters: ["pointer"], result: "void" },
  tach_error: { parameters: ["pointer", "buffer", "usize"], result: "i32" },
} as const;

class Writer {
  readonly #bytes: number[] = [];
  u32(value: number): void {
    this.#bytes.push(
      value & 255,
      value >>> 8 & 255,
      value >>> 16 & 255,
      value >>> 24 & 255,
    );
  }
  raw(value: Uint8Array): void {
    this.u32(value.byteLength);
    this.#bytes.push(...value);
    while (this.#bytes.length % 4) this.#bytes.push(0);
  }
  text(value: string): void {
    this.raw(textEncoder.encode(value));
  }
  finish(): Uint8Array {
    return Uint8Array.from(this.#bytes);
  }
}

function deno(): DenoAPI {
  const runtime = (globalThis as { readonly Deno?: DenoAPI }).Deno;
  if (!runtime) {
    throw new TachError(
      "vulkan-unavailable",
      "Deno is required for Tach's Vulkan host",
      { operation: "tach" },
    );
  }
  return runtime;
}

function libraryURL(runtime: DenoAPI): URL {
  const platform = runtime.build.os === "windows"
    ? ["windows", "x86_64", "dll"]
    : runtime.build.os === "linux"
    ? ["linux", runtime.build.arch, "so"]
    : undefined;
  if (!platform || runtime.build.arch !== "x86_64") {
    throw new TachError(
      "vulkan-unavailable",
      `unsupported Vulkan host ${runtime.build.os}/${runtime.build.arch}`,
      { operation: "tach" },
    );
  }
  return new URL(
    `../native/tach-vulkan.${platform.join(".")}`,
    import.meta.url,
  );
}

function handle(value: DriverBuffer): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new TypeError("invalid Vulkan buffer");
  }
  return value;
}

function moduleDescription(command: PreparedCommand): Uint8Array {
  const target = command.target;
  if (
    target.vulkan !== "1.3" || target.spirv !== "1.6" ||
    !target.features?.includes("synchronization2") ||
    !target.features.includes("shaderZeroInitializeWorkgroupMemory")
  ) {
    throw new TachError(
      "vulkan-profile",
      "generated module does not target Vulkan 1.3 and SPIR-V 1.6",
      { operation: "module" },
    );
  }
  const output = new Writer();
  output.u32(abi);
  output.u32(target.kernels.length);
  for (const kernel of target.kernels) {
    output.text(kernel.entryPoint);
    output.u32(kernel.bindings.length);
    for (const binding of kernel.bindings) {
      output.u32(binding.binding);
      output.u32(binding.minimumByteSize);
    }
    output.u32(kernel.parameterBlock ? 1 : 0);
    if (kernel.parameterBlock) {
      output.u32(kernel.parameterBlock.binding);
      output.u32(kernel.parameterBlock.byteSize);
    }
  }
  return output.finish();
}

function resource(
  output: Writer,
  value: PreparedResource | PreparedBarrierResource,
): void {
  if (value.buffer !== undefined) {
    output.u32(0);
    output.u32(handle(value.buffer));
  } else {
    output.u32(1);
    output.u32(value.scratch!);
  }
}

class VulkanDriver implements Driver {
  readonly adapter: TachAdapterInfo;
  readonly #modules = new WeakMap<ModuleDefinition, Promise<number>>();
  readonly #pending = new Set<bigint>();
  #closed = false;

  constructor(
    readonly runtime: DenoAPI,
    readonly library: ReturnType<DenoAPI["dlopen"]>,
    readonly session: Pointer,
    adapter: TachAdapterInfo,
  ) {
    this.adapter = adapter;
  }

  #call(
    name: keyof typeof symbols,
    ...arguments_: readonly unknown[]
  ): number | Promise<number> {
    if (this.#closed) {
      throw new TachError("lifecycle", "Vulkan driver is closed", {
        operation: name,
      });
    }
    return this.library.symbols[name]!(...arguments_);
  }

  #error(operation: string, status: number): never {
    const bytes = new Uint8Array(4096),
      length = this.library.symbols.tach_error!(
        this.session,
        bytes,
        bytes.byteLength,
      ) as number;
    throw new TachError(
      "native",
      length > 0
        ? textDecoder.decode(bytes.subarray(0, length))
        : `native Vulkan operation failed (${status})`,
      { operation },
    );
  }

  #check(operation: string, status: number): void {
    if (status !== 0) this.#error(operation, status);
  }

  createBuffer(_label: string, bytes: Uint8Array): DriverBuffer {
    const output = new Uint32Array(1),
      status = this.#call(
        "tach_buffer",
        this.session,
        bytes.byteLength,
        bytes,
        output,
      ) as number;
    this.#check("buffer", status);
    return output[0]!;
  }
  writeBuffer(buffer: DriverBuffer, bytes: Uint8Array): void {
    this.#check(
      "buffer.write",
      this.#call(
        "tach_write",
        this.session,
        handle(buffer),
        bytes,
        bytes.byteLength,
      ) as number,
    );
  }
  async readBuffer(
    buffer: DriverBuffer,
    byteLength: number,
  ): Promise<Uint8Array> {
    const output = new Uint8Array(byteLength),
      status = await this.#call(
        "tach_read",
        this.session,
        handle(buffer),
        output,
        byteLength,
      );
    this.#check("buffer.read", status);
    return output;
  }
  destroyBuffer(buffer: DriverBuffer): void {
    this.#call("tach_destroy_buffer", this.session, handle(buffer));
  }

  #module(command: PreparedCommand): Promise<number> {
    let pending = this.#modules.get(command.module);
    if (!pending) {
      pending = (async () => {
        if (command.shader.protocol !== "file:") {
          throw new TachError(
            "vulkan-profile",
            "Vulkan modules require a local SPIR-V URL",
            { operation: "module" },
          );
        }
        const spirv = await this.runtime.readFile(command.shader),
          description = moduleDescription(command),
          output = new Uint32Array(1);
        const status = await this.#call(
          "tach_module",
          this.session,
          spirv,
          spirv.byteLength,
          description,
          description.byteLength,
          output,
        );
        this.#check("module", status);
        return output[0]!;
      })();
      this.#modules.set(command.module, pending);
    }
    return pending;
  }

  async submit(commands: readonly PreparedCommand[]): Promise<void> {
    const modules = await Promise.all(
        commands.map((command) => this.#module(command)),
      ),
      output = new Writer();
    output.u32(abi);
    output.u32(commands.length);
    commands.forEach((command, commandIndex) => {
      output.u32(modules[commandIndex]!);
      output.u32(command.repeat);
      output.u32(command.scratch.size);
      for (const [color, bytes] of command.scratch) {
        output.u32(color);
        output.u32(bytes);
      }
      output.u32(command.steps.length);
      for (const step of command.steps) {
        output.u32(step.kind === "dispatch" ? 0 : 1);
        if (step.kind === "barrier") {
          output.u32(step.resources.length);
          for (const value of step.resources) resource(output, value);
          continue;
        }
        output.u32(step.kernel);
        for (const group of step.groups) output.u32(group);
        output.u32(step.resources.length);
        for (const value of step.resources) {
          output.u32(value.binding);
          resource(output, value);
          output.u32(value.byteSize);
        }
        output.raw(step.parameters ?? new Uint8Array());
      }
      output.u32(command.repeatBarrier?.length ?? 0);
      for (const value of command.repeatBarrier ?? []) resource(output, value);
    });
    const batch = output.finish(),
      submission = new BigUint64Array(1),
      status = await this.#call(
        "tach_submit",
        this.session,
        batch,
        batch.byteLength,
        submission,
      );
    this.#check("submit", status);
    this.#pending.add(submission[0]!);
  }

  async idle(): Promise<void> {
    for (const submission of this.#pending) {
      this.#check(
        "idle",
        await this.#call("tach_wait", this.session, submission),
      );
      this.#pending.delete(submission);
    }
  }

  close(): void {
    if (this.#closed) return;
    this.library.symbols.tach_close!(this.session);
    this.#closed = true;
  }
}

export function openDeno(options: TachOptions): Promise<Driver> {
  const runtime = deno();
  let library: ReturnType<DenoAPI["dlopen"]>;
  try {
    library = nativeLibrary ??= runtime.dlopen(libraryURL(runtime), symbols);
  } catch (cause) {
    throw new TachError(
      "vulkan-unavailable",
      "could not load Tach's Vulkan runtime",
      { operation: "tach", cause },
    );
  }
  if (library.symbols.tach_abi_version!() !== abi) {
    throw new TachError("vulkan-profile", "native Tach ABI version mismatch", {
      operation: "tach",
    });
  }
  const encoded = textEncoder.encode(JSON.stringify(options)),
    pointer = new BigUint64Array(1),
    status = library.symbols.tach_open!(
      encoded,
      encoded.byteLength,
      pointer,
    ) as number;
  const session = runtime.UnsafePointer.create(pointer[0]!);
  if (status !== 0 || !session) {
    const bytes = new Uint8Array(4096),
      length = library.symbols.tach_error!(
        null,
        bytes,
        bytes.byteLength,
      ) as number;
    throw new TachError(
      "vulkan-unavailable",
      length > 0
        ? textDecoder.decode(bytes.subarray(0, length))
        : `could not open Vulkan (${status})`,
      { operation: "tach" },
    );
  }
  const bytes = new Uint8Array(4096),
    length = library.symbols.tach_info!(
      session,
      bytes,
      bytes.byteLength,
    ) as number;
  if (length <= 0) {
    library.symbols.tach_close!(session);
    throw new TachError("native", "could not read Vulkan adapter information", {
      operation: "tach",
    });
  }
  return Promise.resolve(
    new VulkanDriver(
      runtime,
      library,
      session,
      JSON.parse(
        textDecoder.decode(bytes.subarray(0, length)),
      ) as TachAdapterInfo,
    ),
  );
}

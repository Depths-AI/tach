export interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}

declare const computeCommandBrand: unique symbol;
export interface ComputeCommand {
  readonly [computeCommandBrand]: never;
}
export type LaunchSize =
  | number
  | readonly [x: number, y: number]
  | readonly [x: number, y: number, z: number];
export interface CommandOptions {
  readonly repeat?: number;
}
export interface LaunchOptions<Size extends LaunchSize = LaunchSize>
  extends CommandOptions {
  readonly size?: Size;
}
export interface TachOptions {
  readonly powerPreference?: "low-power" | "high-performance";
}
export type TachBackend = "webgpu" | "vulkan";
export interface TachAdapterInfo {
  readonly backend: TachBackend;
  readonly name: string;
  readonly vendor?: string;
  readonly architecture?: string;
  readonly type?: "integrated" | "discrete" | "virtual" | "cpu" | "unknown";
}
export interface Tach {
  readonly adapter: TachAdapterInfo;
  buffer<T>(value: T): ComputeBuffer<T>;
  submit(
    first: ComputeCommand,
    ...rest: readonly ComputeCommand[]
  ): Promise<void>;
  idle(): Promise<void>;
  close(): void;
}
export interface TachFunction {
  (options?: TachOptions): Promise<Tach>;
  <T>(work: (gpu: Tach) => T | Promise<T>, options?: TachOptions): Promise<T>;
}

export type TachErrorCode =
  | "webgpu-unavailable"
  | "adapter-unavailable"
  | "device-request-failed"
  | "device-lost"
  | "gpu-validation"
  | "gpu-out-of-memory"
  | "gpu-internal"
  | "vulkan-unavailable"
  | "vulkan-profile"
  | "native"
  | "buffer"
  | "kernel"
  | "lifecycle"
  | "user"
  | "compiler-platform"
  | "compiler-install"
  | "compiler-execution";

export class TachError extends Error {
  readonly code: TachErrorCode;
  readonly operation: string | undefined;

  constructor(
    code: TachErrorCode,
    message: string,
    options: { readonly operation?: string; readonly cause?: unknown } = {},
  ) {
    super(
      message,
      options.cause === undefined ? undefined : { cause: options.cause },
    );
    this.name = "TachError";
    this.code = code;
    this.operation = options.operation;
  }
}

export function normalizeError(
  cause: unknown,
  code: TachErrorCode,
  operation?: string,
): TachError {
  if (cause instanceof TachError) return cause;
  let name: string | undefined, message = "Unknown error";
  try {
    const candidate = cause !== null &&
        (typeof cause === "object" || typeof cause === "function")
      ? cause as {
        readonly constructor?: { readonly name?: string };
        readonly message?: unknown;
      }
      : undefined;
    name = candidate?.constructor?.name;
    message = typeof candidate?.message === "string"
      ? candidate.message
      : String(cause);
  } catch { /* Error normalization must not become a second failure. */ }
  const gpuCode = name === "GPUValidationError"
    ? "gpu-validation"
    : name === "GPUOutOfMemoryError"
    ? "gpu-out-of-memory"
    : name === "GPUInternalError"
    ? "gpu-internal"
    : undefined;
  return new TachError(gpuCode ?? code, message, {
    ...(operation === undefined ? {} : { operation }),
    cause,
  });
}

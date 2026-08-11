export type TachErrorCode =
  | "webgpu-unavailable"
  | "adapter-unavailable"
  | "device-request-failed"
  | "device-lost"
  | "gpu-validation"
  | "gpu-out-of-memory"
  | "gpu-internal"
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
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
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
  let name: string | undefined;
  let message = "Unknown error";
  try {
    const candidate = cause !== null && (typeof cause === "object" || typeof cause === "function")
      ? cause as { readonly constructor?: { readonly name?: string }; readonly message?: unknown }
      : undefined;
    name = candidate?.constructor?.name;
    message = typeof candidate?.message === "string" ? candidate.message : String(cause);
  } catch {
    // Error normalization must not become a second failure.
  }
  const gpuCode = name === "GPUValidationError" ? "gpu-validation"
    : name === "GPUOutOfMemoryError" ? "gpu-out-of-memory"
    : name === "GPUInternalError" ? "gpu-internal"
    : undefined;
  return new TachError(gpuCode ?? code, message, {
    ...(operation === undefined ? {} : { operation }),
    cause,
  });
}

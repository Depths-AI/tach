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

export interface TachError {
  readonly code: TachErrorCode;
  readonly message: string;
  readonly operation?: string;
  readonly cause?: unknown;
}

export type Result<T, E = TachError> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export function ok<T>(value: T): Result<T, never> {
  return Object.freeze({ ok: true, value });
}

export function err<E>(error: E): Result<never, E> {
  return Object.freeze({ ok: false, error });
}

export function tachError(
  code: TachErrorCode,
  message: string,
  options: { readonly operation?: string; readonly cause?: unknown } = {},
): TachError {
  return Object.freeze({
    code,
    message,
    ...(options.operation === undefined ? {} : { operation: options.operation }),
    ...(options.cause === undefined ? {} : { cause: options.cause }),
  });
}

export class TachFailure extends Error {
  readonly data: TachError;

  constructor(data: TachError) {
    super(data.message, data.cause === undefined ? undefined : { cause: data.cause });
    this.name = "TachFailure";
    this.data = data;
  }
}

export function normalizeError(
  cause: unknown,
  code: TachErrorCode,
  operation?: string,
): TachError {
  if (cause instanceof TachFailure) return cause.data;
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
  return tachError(gpuCode ?? code, message, {
    ...(operation === undefined ? {} : { operation }),
    cause,
  });
}

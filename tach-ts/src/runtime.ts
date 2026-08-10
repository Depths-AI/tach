import {
  err,
  normalizeError,
  ok,
  TachFailure,
  tachError,
  type Result,
  type TachErrorCode,
} from "./result.js";

const bufferUsage = {
  mapRead: 0x0001,
  copyDst: 0x0008,
} as const;

const mapMode = {
  read: 0x0001,
} as const;

const noFailure = Symbol();

export interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}

export type DispatchSize = number | readonly [x: number, y?: number, z?: number];

export interface DispatchOptions {
  readonly size?: DispatchSize;
  readonly dispatches?: number;
}

export interface TachOptions {
  readonly gpu?: GPU;
  readonly adapter?: GPURequestAdapterOptions;
  readonly device?: GPUDeviceDescriptor;
}

export interface Tach {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  buffer<T>(value: T): ComputeBuffer<T>;
  close(): void;
}

export interface BufferCodec<T> {
  readonly key: string;
  pack(value: T): Uint8Array;
  unpack(source: ArrayBuffer | ArrayBufferView): T;
}

export interface Submission<T> {
  finish(): Promise<T>;
  cleanup?(): void;
}

export interface RuntimeOwner {
  readonly device: GPUDevice;
  register(state: BufferState<unknown>): void;
  unregister(state: BufferState<unknown>): void;
  assertHealthy(operation: string): void;
  capture<T>(
    operation: string,
    fallbackCode: TachErrorCode,
    issue: () => Submission<T>,
  ): Promise<T>;
}

export interface BufferState<T> {
  readonly owner: RuntimeOwner;
  value: T;
  byteLength: number;
  codec?: BufferCodec<T>;
  destroyed: boolean;
  gpu?: GPUBuffer;
}

const bufferStates = new WeakMap<object, BufferState<unknown>>();

function clone<T>(value: T, operation: string): T {
  try {
    return structuredClone(value);
  } catch (cause) {
    throw new TachFailure(tachError("buffer", "compute buffer data is not cloneable", {
      operation,
      cause,
    }));
  }
}

class Session implements Tach, RuntimeOwner {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;

  readonly #buffers = new Set<BufferState<unknown>>();
  readonly #uncaptured: GPUError[] = [];
  #closed = false;
  #lost?: GPUDeviceLostInfo;

  constructor(adapter: GPUAdapter, device: GPUDevice) {
    this.adapter = adapter;
    this.device = device;
    device.addEventListener("uncapturederror", this.#onUncaptured);
    void device.lost.then((info) => {
      if (!this.#closed) this.#lost = info;
    });
  }

  readonly #onUncaptured = (event: GPUUncapturedErrorEvent): void => {
    event.preventDefault();
    this.#uncaptured.push(event.error);
  };

  buffer<T>(value: T): ComputeBuffer<T> {
    this.assertHealthy("buffer");
    return createComputeBuffer(this, value);
  }

  register(state: BufferState<unknown>): void {
    this.assertHealthy("buffer");
    this.#buffers.add(state);
  }

  unregister(state: BufferState<unknown>): void {
    this.#buffers.delete(state);
  }

  assertHealthy(operation: string): void {
    if (this.#closed) {
      throw new TachFailure(tachError("lifecycle", "Tach session is closed", { operation }));
    }
    if (this.#lost) {
      throw new TachFailure(tachError(
        "device-lost",
        this.#lost.message || `GPU device was lost (${this.#lost.reason})`,
        { operation, cause: this.#lost },
      ));
    }
    const uncaptured = this.#uncaptured.shift();
    if (uncaptured) {
      throw new TachFailure(normalizeError(uncaptured, "gpu-internal", operation));
    }
  }

  async capture<T>(
    operation: string,
    fallbackCode: TachErrorCode,
    issue: () => Submission<T>,
  ): Promise<T> {
    this.assertHealthy(operation);
    this.device.pushErrorScope("internal");
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");

    let submission: Submission<T> | undefined;
    let issueError: unknown = noFailure;
    try {
      submission = issue();
    } catch (cause) {
      issueError = cause;
    }

    // Popping immediately after the synchronous WebGPU calls keeps concurrent
    // kernel invocations from interleaving their error-scope stacks.
    const validation = this.device.popErrorScope();
    const outOfMemory = this.device.popErrorScope();
    const internal = this.device.popErrorScope();

    let value: T | undefined;
    let finishError: unknown = noFailure;
    if (submission) {
      try {
        value = await submission.finish();
      } catch (cause) {
        finishError = cause;
      }
    }

    let scopeErrors: readonly (GPUError | null)[] = [];
    let scopeFailure: unknown;
    try {
      scopeErrors = await Promise.all([validation, outOfMemory, internal]);
    } catch (cause) {
      scopeFailure = cause;
    } finally {
      try {
        submission?.cleanup?.();
      } catch (cause) {
        finishError ??= cause;
      }
    }

    if (issueError !== noFailure) {
      throw new TachFailure(normalizeError(issueError, fallbackCode, operation));
    }
    if (finishError !== noFailure) {
      throw new TachFailure(normalizeError(finishError, fallbackCode, operation));
    }
    if (scopeFailure !== undefined) {
      throw new TachFailure(normalizeError(scopeFailure, "gpu-internal", operation));
    }
    const scoped = scopeErrors.find((error): error is GPUError => error !== null);
    if (scoped) {
      throw new TachFailure(normalizeError(scoped, "gpu-internal", operation));
    }
    this.assertHealthy(operation);
    return value as T;
  }

  async settle(operation: string): Promise<void> {
    this.assertHealthy(operation);
    try {
      await this.device.queue.onSubmittedWorkDone();
    } catch (cause) {
      throw new TachFailure(normalizeError(cause, "device-lost", operation));
    }
    this.assertHealthy(operation);
  }

  close(): void {
    if (this.#closed) return;
    for (const state of [...this.#buffers]) destroyBufferState(state);
    this.device.removeEventListener("uncapturederror", this.#onUncaptured);
    this.#closed = true;
    this.device.destroy();
  }
}

function createComputeBuffer<T>(owner: Session, initial: T): ComputeBuffer<T> {
  const state: BufferState<T> = {
    owner,
    value: clone(initial, "buffer"),
    byteLength: 0,
    destroyed: false,
  };
  const handle: ComputeBuffer<T> = {
    write(value) {
      live(state, "buffer.write");
      const next = clone(value, "buffer.write");
      if (!state.gpu || !state.codec) {
        state.value = next;
        return;
      }
      try {
        const bytes = state.codec.pack(next);
        if (bytes.byteLength !== state.byteLength) {
          throw new RangeError("compute buffers cannot change size; create a new buffer instead");
        }
        state.owner.device.queue.writeBuffer(
          state.gpu,
          0,
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength,
        );
        state.value = next;
      } catch (cause) {
        throw new TachFailure(normalizeError(cause, "buffer", "buffer.write"));
      }
    },
    async read() {
      live(state, "buffer.read");
      if (!state.gpu || !state.codec) return clone(state.value, "buffer.read");

      return state.owner.capture("buffer.read", "buffer", () => {
        const readback = state.owner.device.createBuffer({
          label: "Tach readback",
          size: Math.max(4, align4(state.byteLength)),
          usage: bufferUsage.copyDst | bufferUsage.mapRead,
        });
        let mapped = false;
        try {
          const encoder = state.owner.device.createCommandEncoder({
            label: "Tach readback commands",
          });
          encoder.copyBufferToBuffer(state.gpu as GPUBuffer, 0, readback, 0, state.byteLength);
          state.owner.device.queue.submit([encoder.finish()]);
          return {
            async finish() {
              await readback.mapAsync(mapMode.read);
              mapped = true;
              const bytes = new Uint8Array(readback.getMappedRange()).slice(0, state.byteLength);
              state.value = (state.codec as BufferCodec<T>).unpack(bytes);
              return clone(state.value, "buffer.read");
            },
            cleanup() {
              if (mapped) readback.unmap();
              readback.destroy();
            },
          };
        } catch (cause) {
          readback.destroy();
          throw cause;
        }
      });
    },
    destroy() {
      destroyBufferState(state);
    },
  };
  bufferStates.set(handle, state as BufferState<unknown>);
  owner.register(state as BufferState<unknown>);
  return Object.freeze(handle);
}

function live<T>(state: BufferState<T>, operation: string): void {
  state.owner.assertHealthy(operation);
  if (state.destroyed) {
    throw new TachFailure(tachError("lifecycle", "compute buffer has been destroyed", {
      operation,
    }));
  }
}

function destroyBufferState(state: BufferState<unknown>): void {
  if (state.destroyed) return;
  state.destroyed = true;
  state.gpu?.destroy();
  delete state.gpu;
  state.owner.unregister(state);
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

export function getBufferState<T>(
  value: ComputeBuffer<T>,
  name: string,
): BufferState<T> {
  const state = bufferStates.get(value) as BufferState<T> | undefined;
  if (!state) {
    throw new TachFailure(tachError(
      "buffer",
      `${name} must be created by Tach.buffer(value)`,
      { operation: name },
    ));
  }
  live(state, name);
  return state;
}

export async function openTach(options: TachOptions = {}): Promise<Result<Tach>> {
  const gpu = options.gpu ?? (typeof navigator === "undefined" ? undefined : navigator.gpu);
  if (!gpu) {
    return err(tachError(
      "webgpu-unavailable",
      "WebGPU is unavailable in this environment",
      { operation: "openTach" },
    ));
  }

  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter(options.adapter);
  } catch (cause) {
    return err(normalizeError(cause, "adapter-unavailable", "requestAdapter"));
  }
  if (!adapter) {
    return err(tachError(
      "adapter-unavailable",
      "WebGPU did not provide an adapter",
      { operation: "requestAdapter" },
    ));
  }

  try {
    const device = await adapter.requestDevice(options.device);
    return ok(new Session(adapter, device));
  } catch (cause) {
    return err(normalizeError(cause, "device-request-failed", "requestDevice"));
  }
}

export async function tach<T>(
  work: (gpu: Tach) => T | Promise<T>,
  options: TachOptions = {},
): Promise<Result<T>> {
  const opened = await openTach(options);
  if (!opened.ok) return opened;
  const session = opened.value as Session;

  let result: Result<T>;
  try {
    const value = await work(session);
    await session.settle("tach");
    result = ok(value);
  } catch (cause) {
    result = err(normalizeError(cause, "user", "tach"));
  }

  try {
    session.close();
  } catch (cause) {
    if (result.ok) result = err(normalizeError(cause, "lifecycle", "close"));
  }
  return result;
}

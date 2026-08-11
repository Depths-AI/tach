import {
  err,
  normalizeError,
  ok,
  TachFailure,
  tachError,
  type Result,
  type TachError,
  type TachErrorCode,
} from "./result.js";

const bufferUsage = {
  mapRead: 0x0001,
  copyDst: 0x0008,
  uniform: 0x0040,
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

declare const computeDispatchBrand: unique symbol;

export interface ComputeDispatch {
  readonly [computeDispatchBrand]: never;
}

export type DispatchSize = number | readonly [x: number, y: number] | readonly [x: number, y: number, z: number];

export interface DispatchOptions<Size extends DispatchSize = DispatchSize> {
  readonly size?: Size;
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
  submit(first: ComputeDispatch, ...rest: readonly ComputeDispatch[]): Promise<void>;
  idle(): Promise<void>;
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

export interface BufferBindGroupEntry {
  readonly binding: number;
  readonly resource: GPUBufferBinding;
}

export interface PreparedDispatch {
  readonly uniforms: readonly Uint8Array[];
  encode(
    pass: GPUComputePassEncoder,
    uniforms: readonly GPUBufferBinding[],
  ): void;
}

export interface DispatchState {
  readonly owner: RuntimeOwner;
  prepare(): Promise<PreparedDispatch>;
}

export interface RuntimeOwner {
  readonly device: GPUDevice;
  register(state: BufferState<unknown>): void;
  unregister(state: BufferState<unknown>): void;
  assertHealthy(operation: string): void;
  waitForSubmissions(operation: string): Promise<void>;
  bindGroup(
    label: string,
    layout: GPUBindGroupLayout,
    entries: readonly BufferBindGroupEntry[],
  ): GPUBindGroup;
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
const dispatchStates = new WeakMap<object, DispatchState>();

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
  // DECISION: Cache live binding sets; use a bounded LRU if high-churn sessions make combinations unbounded.
  readonly #bindGroups = new Map<string, GPUBindGroup>();
  readonly #objectIDs = new WeakMap<object, number>();
  readonly #checks = new Set<Promise<void>>();
  readonly #uncaptured: GPUError[] = [];
  #nextObjectID = 1;
  #submissionTail: Promise<void> = Promise.resolve();
  #deferredFailure?: TachError;
  #uniform: GPUBuffer | undefined;
  #uniformCapacity = 0;
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

  submit(first: ComputeDispatch, ...rest: readonly ComputeDispatch[]): Promise<void> {
    this.assertHealthy("submit");
    const values = [first, ...rest];
    const states = values.map((value, index) => getDispatchState(value, `submit[${index}]`));
    for (const state of states) {
      if (state.owner !== this) {
        throw new TachFailure(tachError(
          "lifecycle",
          "compute dispatch belongs to a different Tach session",
          { operation: "submit" },
        ));
      }
    }

    const pending = this.#submissionTail.then(async () => {
      this.assertHealthy("submit");
      const prepared = await Promise.all(states.map((state) => state.prepare()));
      this.assertHealthy("submit");
      this.#record(prepared);
    });
    this.#submissionTail = pending.then(
      () => undefined,
      (cause) => {
        this.#deferredFailure ??= normalizeError(cause, "kernel", "submit");
      },
    );
    return pending;
  }

  async idle(): Promise<void> {
    await this.waitForSubmissions("idle");
    try {
      await this.device.queue.onSubmittedWorkDone();
      await Promise.all([...this.#checks]);
    } catch (cause) {
      throw new TachFailure(normalizeError(cause, "device-lost", "idle"));
    }
    this.assertHealthy("idle");
  }

  register(state: BufferState<unknown>): void {
    this.assertHealthy("buffer");
    this.#buffers.add(state);
  }

  unregister(state: BufferState<unknown>): void {
    this.#buffers.delete(state);
    this.#bindGroups.clear();
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
    if (this.#deferredFailure) {
      throw new TachFailure(this.#deferredFailure);
    }
    const uncaptured = this.#uncaptured.shift();
    if (uncaptured) {
      throw new TachFailure(normalizeError(uncaptured, "gpu-internal", operation));
    }
  }

  async waitForSubmissions(operation: string): Promise<void> {
    this.assertHealthy(operation);
    await this.#submissionTail;
    this.assertHealthy(operation);
  }

  bindGroup(
    label: string,
    layout: GPUBindGroupLayout,
    entries: readonly BufferBindGroupEntry[],
  ): GPUBindGroup {
    const key = [
      this.#objectID(layout),
      ...entries.map((entry) => [
        entry.binding,
        this.#objectID(entry.resource.buffer),
        entry.resource.offset ?? 0,
        entry.resource.size ?? entry.resource.buffer.size,
      ].join(":")),
    ].join("|");
    let group = this.#bindGroups.get(key);
    if (!group) {
      group = this.device.createBindGroup({ label, layout, entries: [...entries] });
      this.#bindGroups.set(key, group);
    }
    return group;
  }

  #objectID(value: object): number {
    let id = this.#objectIDs.get(value);
    if (id === undefined) {
      id = this.#nextObjectID++;
      this.#objectIDs.set(value, id);
    }
    return id;
  }

  #record(dispatches: readonly PreparedDispatch[]): void {
    this.#captureDeferred("submit", "kernel", () => {
      const uniformBindings = this.#writeUniforms(dispatches);
      const encoder = this.device.createCommandEncoder({ label: "Tach submission" });
      const pass = encoder.beginComputePass({ label: "Tach compute pass" });
      let uniformIndex = 0;
      for (const dispatch of dispatches) {
        const next = uniformIndex + dispatch.uniforms.length;
        dispatch.encode(pass, uniformBindings.slice(uniformIndex, next));
        uniformIndex = next;
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    });
  }

  #writeUniforms(dispatches: readonly PreparedDispatch[]): readonly GPUBufferBinding[] {
    const alignment = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    const chunks = dispatches.flatMap((dispatch) => dispatch.uniforms);
    let byteLength = 0;
    const offsets = chunks.map((bytes) => {
      byteLength = align(byteLength, alignment);
      const offset = byteLength;
      byteLength += bytes.byteLength;
      return offset;
    });
    if (byteLength === 0) return [];

    const buffer = this.#uniformBuffer(byteLength);
    const upload = new Uint8Array(byteLength);
    for (let index = 0; index < chunks.length; index++) {
      upload.set(chunks[index] as Uint8Array, offsets[index]);
    }
    this.device.queue.writeBuffer(buffer, 0, upload.buffer, upload.byteOffset, upload.byteLength);
    return chunks.map((bytes, index) => ({
      buffer,
      offset: offsets[index]!,
      size: bytes.byteLength,
    }));
  }

  #uniformBuffer(byteLength: number): GPUBuffer {
    if (this.#uniform && this.#uniformCapacity >= byteLength) return this.#uniform;
    let capacity = Math.max(4096, this.#uniformCapacity);
    while (capacity < byteLength) capacity *= 2;
    if (!Number.isSafeInteger(capacity)) {
      throw new RangeError("uniform data exceeds JavaScript's safe integer range");
    }
    const next = this.device.createBuffer({
      label: "Tach uniform arena",
      size: align(capacity, 4),
      usage: bufferUsage.copyDst | bufferUsage.uniform,
    });
    this.#uniform?.destroy();
    this.#uniform = next;
    this.#uniformCapacity = capacity;
    this.#bindGroups.clear();
    return next;
  }

  #captureDeferred(
    operation: string,
    fallbackCode: TachErrorCode,
    issue: () => void,
  ): void {
    this.assertHealthy(operation);
    this.device.pushErrorScope("internal");
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");

    let issueError: unknown = noFailure;
    try {
      issue();
    } catch (cause) {
      issueError = cause;
    }

    const validation = this.device.popErrorScope();
    const outOfMemory = this.device.popErrorScope();
    const internal = this.device.popErrorScope();
    let check!: Promise<void>;
    check = Promise.all([validation, outOfMemory, internal])
      .then((errors) => {
        const scoped = errors.find((error): error is GPUError => error !== null);
        if (scoped) this.#deferredFailure ??= normalizeError(scoped, fallbackCode, operation);
      }, (cause) => {
        this.#deferredFailure ??= normalizeError(cause, "gpu-internal", operation);
      })
      .finally(() => this.#checks.delete(check));
    this.#checks.add(check);

    if (issueError !== noFailure) {
      throw new TachFailure(normalizeError(issueError, fallbackCode, operation));
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

  close(): void {
    if (this.#closed) return;
    for (const state of [...this.#buffers]) destroyBufferState(state);
    this.#uniform?.destroy();
    this.#uniform = undefined;
    this.#bindGroups.clear();
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
      await state.owner.waitForSubmissions("buffer.read");
      live(state, "buffer.read");
      if (!state.gpu || !state.codec) return clone(state.value, "buffer.read");

      return state.owner.capture("buffer.read", "buffer", () => {
        const readback = state.owner.device.createBuffer({
          label: "Tach readback",
          size: Math.max(4, align(state.byteLength, 4)),
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

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
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

export function createComputeDispatch(state: DispatchState): ComputeDispatch {
  const handle = Object.freeze({
    then(): never {
      throw new TachFailure(tachError(
        "kernel",
        "compute dispatches must be passed to Tach.submit()",
        { operation: "dispatch" },
      ));
    },
  }) as unknown as ComputeDispatch;
  dispatchStates.set(handle, state);
  return handle;
}

function getDispatchState(value: ComputeDispatch, operation: string): DispatchState {
  const state = value && typeof value === "object" ? dispatchStates.get(value) : undefined;
  if (!state) {
    throw new TachFailure(tachError(
      "kernel",
      `${operation} must be a generated Tach compute dispatch`,
      { operation: "submit" },
    ));
  }
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
    await session.idle();
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

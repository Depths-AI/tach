import type { Driver, DriverBuffer, PreparedCommand } from "./driver.ts";
import { normalizeError, TachError } from "./api.ts";
import type {
  ComputeBuffer,
  ComputeCommand,
  PresentationCanvas,
  Tach,
  TachAdapterInfo,
  TachFunction,
  TachOptions,
} from "./api.ts";

const noFailure = Symbol();

export interface BufferCodec<T> {
  readonly key: string;
  pack(value: T): Uint8Array;
  unpack(source: ArrayBuffer | ArrayBufferView): T;
}
export interface BufferState<T> {
  readonly owner: RuntimeOwner;
  value: T;
  byteLength: number;
  codec?: BufferCodec<T>;
  destroyed: boolean;
  driverBuffer?: DriverBuffer;
}
export interface CommandState {
  readonly owner: RuntimeOwner;
  prepare(): PreparedCommand;
}
export interface RuntimeOwner {
  readonly driver: Driver;
  register(state: BufferState<unknown>): void;
  unregister(state: BufferState<unknown>): void;
  assertHealthy(operation: string): void;
  waitForSubmissions(operation: string): Promise<void>;
}

const bufferStates = new WeakMap<object, BufferState<unknown>>();
const commandStates = new WeakMap<object, CommandState>();

function clone<T>(value: T, operation: string): T {
  try {
    return structuredClone(value);
  } catch (cause) {
    throw new TachError("buffer", "compute buffer data is not cloneable", {
      operation,
      cause,
    });
  }
}

class Session implements Tach, RuntimeOwner {
  readonly adapter: TachAdapterInfo;
  readonly #buffers = new Set<BufferState<unknown>>();
  #submissionTail: Promise<void> = Promise.resolve();
  #deferredFailure?: TachError;
  #presenting = false;
  #closed = false;

  constructor(readonly driver: Driver) {
    this.adapter = driver.adapter;
  }

  buffer<T>(value: T): ComputeBuffer<T> {
    this.assertHealthy("buffer");
    return createComputeBuffer(this, value);
  }

  #execute(
    operation: "prepare" | "submit" | "present",
    first: ComputeCommand,
    rest: readonly ComputeCommand[],
    presentation?: {
      readonly canvas: PresentationCanvas;
      readonly pixels: BufferState<Uint32Array | readonly number[]>;
    },
  ): Promise<void> {
    this.assertHealthy(operation);
    const states = [first, ...rest].map((value, index) =>
      getCommandState(value, `${operation}[${index}]`)
    );
    for (const state of states) {
      if (state.owner !== this) {
        throw new TachError(
          "lifecycle",
          "compute command belongs to a different Tach session",
          { operation },
        );
      }
    }
    if (presentation && presentation.pixels.owner !== this) {
      throw new TachError(
        "lifecycle",
        "presentation buffer belongs to a different Tach session",
        { operation },
      );
    }
    const pending = this.#submissionTail.then(async () => {
      this.assertHealthy(operation);
      const commands = states.map((state) => state.prepare());
      if (operation === "present") {
        const pixels = presentation!.pixels;
        if (pixels.driverBuffer === undefined || !pixels.codec) {
          throw new TachError(
            "buffer",
            "presentation buffer must be materialized by a submitted command",
            { operation },
          );
        }
        await this.driver.present!(
          commands,
          presentation!.canvas,
          pixels.driverBuffer,
          pixels.byteLength,
        );
      } else {
        await this.driver[operation](commands);
      }
      this.assertHealthy(operation);
    });
    this.#submissionTail = pending.catch((cause) => {
      this.#deferredFailure ??= normalizeError(cause, "kernel", operation);
    });
    return pending;
  }

  prepare(
    first: ComputeCommand,
    ...rest: readonly ComputeCommand[]
  ): Promise<void> {
    return this.#execute("prepare", first, rest);
  }

  submit(
    first: ComputeCommand,
    ...rest: readonly ComputeCommand[]
  ): Promise<void> {
    return this.#execute("submit", first, rest);
  }

  present(
    canvas: PresentationCanvas,
    pixels: ComputeBuffer<Uint32Array | readonly number[]>,
    first: ComputeCommand,
    ...rest: readonly ComputeCommand[]
  ): Promise<void> {
    this.assertHealthy("present");
    if (!this.driver.present) {
      throw new TachError(
        "webgpu-unavailable",
        "GPU presentation is available only in browsers",
        { operation: "present" },
      );
    }
    if (this.#presenting) {
      throw new TachError(
        "lifecycle",
        "another presentation is already pending",
        { operation: "present" },
      );
    }
    const pending = this.#execute("present", first, rest, {
      canvas,
      pixels: getBufferState(pixels, "present.pixels"),
    });
    this.#presenting = true;
    return pending.finally(() => {
      this.#presenting = false;
    });
  }

  async idle(): Promise<void> {
    await this.waitForSubmissions("idle");
    try {
      await this.driver.idle();
    } catch (cause) {
      throw normalizeError(cause, "device-lost", "idle");
    }
    this.assertHealthy("idle");
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
      throw new TachError("lifecycle", "Tach session is closed", { operation });
    }
    if (this.#deferredFailure) throw this.#deferredFailure;
  }

  async waitForSubmissions(operation: string): Promise<void> {
    this.assertHealthy(operation);
    await this.#submissionTail;
    this.assertHealthy(operation);
  }

  close(): void {
    if (this.#closed) return;
    for (const state of [...this.#buffers]) destroyBufferState(state);
    this.driver.close();
    this.#closed = true;
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
      if (state.driverBuffer === undefined || !state.codec) {
        state.value = next;
        return;
      }
      try {
        const bytes = state.codec.pack(next);
        if (bytes.byteLength !== state.byteLength) {
          throw new RangeError(
            "compute buffers cannot change size; create a new buffer instead",
          );
        }
        state.owner.driver.writeBuffer(state.driverBuffer, bytes);
        state.value = next;
      } catch (cause) {
        throw normalizeError(cause, "buffer", "buffer.write");
      }
    },
    async read() {
      live(state, "buffer.read");
      await state.owner.waitForSubmissions("buffer.read");
      live(state, "buffer.read");
      if (state.driverBuffer === undefined || !state.codec) {
        return clone(state.value, "buffer.read");
      }
      try {
        const bytes = await state.owner.driver.readBuffer(
          state.driverBuffer,
          state.byteLength,
        );
        state.value = state.codec.unpack(bytes);
        return clone(state.value, "buffer.read");
      } catch (cause) {
        throw normalizeError(cause, "buffer", "buffer.read");
      }
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
    throw new TachError("lifecycle", "compute buffer has been destroyed", {
      operation,
    });
  }
}

function destroyBufferState(state: BufferState<unknown>): void {
  if (state.destroyed) return;
  state.destroyed = true;
  if (state.driverBuffer !== undefined) {
    state.owner.driver.destroyBuffer(state.driverBuffer);
  }
  delete state.driverBuffer;
  state.owner.unregister(state);
}

export function getBufferState<T>(
  value: ComputeBuffer<T>,
  name: string,
): BufferState<T> {
  const state = bufferStates.get(value) as BufferState<T> | undefined;
  if (!state) {
    throw new TachError(
      "buffer",
      `${name} must be created by Tach.buffer(value)`,
      { operation: name },
    );
  }
  live(state, name);
  return state;
}

export function createComputeCommand(state: CommandState): ComputeCommand {
  const handle = Object.freeze({
    then(): never {
      throw new TachError(
        "kernel",
        "compute commands must be passed to Tach.submit()",
        { operation: "command" },
      );
    },
  }) as unknown as ComputeCommand;
  commandStates.set(handle, state);
  return handle;
}

function getCommandState(
  value: ComputeCommand,
  operation: string,
): CommandState {
  const state = value && typeof value === "object"
    ? commandStates.get(value)
    : undefined;
  if (!state) {
    throw new TachError(
      "kernel",
      `${operation} must be a generated Tach compute command`,
      { operation: "submit" },
    );
  }
  return state;
}

export function createTach(
  open: (options: TachOptions) => Promise<Driver>,
): TachFunction {
  return (async <T>(
    workOrOptions: TachOptions | ((gpu: Tach) => T | Promise<T>) = {},
    options: TachOptions = {},
  ) => {
    if (typeof workOrOptions !== "function") {
      return new Session(await open(workOrOptions));
    }
    const session = new Session(await open(options));
    let value: T | undefined;
    let failure: unknown = noFailure;
    try {
      value = await workOrOptions(session);
      await session.idle();
    } catch (cause) {
      failure = normalizeError(cause, "user", "tach");
    }
    try {
      session.close();
    } catch (cause) {
      if (failure === noFailure) {
        failure = normalizeError(cause, "lifecycle", "close");
      }
    }
    if (failure !== noFailure) throw failure;
    return value as T;
  }) as TachFunction;
}

export { TachError, type TachErrorCode } from "./api.ts";
export type {
  CommandOptions,
  ComputeBuffer,
  ComputeCommand,
  ComputeView,
  LaunchOptions,
  LaunchSize,
  PresentationCanvas,
  Tach,
  TachAdapterInfo,
  TachBackend,
  TachFunction,
  TachOptions,
} from "./api.ts";
import type { TachFunction, TachOptions } from "./api.ts";
import { createTach } from "./runtime.ts";

async function openHost(options: TachOptions) {
  return "Deno" in globalThis
    ? (await import("./deno.ts")).openDeno(options)
    : (await import("./web.ts")).openWeb(options);
}

export const tach: TachFunction = createTach(openHost);

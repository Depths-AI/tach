# Tach browser correctness harness

This private npm workspace proves that the canonical `examples/` Tach project
works through the published `@depths/tach` surface in Chromium WebGPU. The
harness is Deno-native, standalone, and owns its project build, static server,
Chrome launch, DevTools Protocol client, and semantic assertions. It shares no
runner helper or handwritten shader fixture with another workspace.

## What it exercises

`scripts/build.ts` calls `@depths/tach/compiler` on `examples/`, copies the
complete generated package into local ignored `generated/`, bundles
`src/main.ts`, and places the exact generated WGSL beside the browser module.
The browser imports all public commands through one generated `index.js`.

The seven canonical programs cover:

- shared-memory atomics;
- shifts, masks, complements, and unsigned integer behavior;
- structured branches and compound assignment;
- bounded `for` lowering;
- vectors and the complete scalar math intrinsic set;
- imported struct types and particle integration; and
- scalar runtime arrays, repeated commands, and parameter isolation.

One scoped Tach session submits the complete corpus. Assertions use decoded
GPU results, including exact integer outcomes and tolerance-checked floating
point results. Batched commands with different uniform values and later
submissions verify ordering and parameter-arena separation. Success therefore
proves project discovery/imports, the global facade, multi-entry WGSL,
generated codecs, public command construction, execution, readback, and
cleanup together.

## Browser runner

`test.ts` starts a loopback-only Deno server and launches an installed Chrome
or Chromium with WebGPU enabled. It uses the browser's DevTools Protocol
directly and has no external runner or bundler dependency.
Set `CHROME_BIN` only when Chrome is outside the standard platform locations.

The server exposes only its local app bundle and `kernel.wgsl`. The runner
waits for a single browser result, prints the selected adapter and number of
programs, then closes Chrome, aborts the server, and removes its temporary
profile even after failure.

## Run

From the repository root:

```sh
npm ci --ignore-scripts
npm test --workspace=@tach/browser-test
```

The workspace depends on local `@depths/tach` through the root npm workspace.
`npm run build` must have produced `dist/tach.exe` or `dist/tach`; alternatively
`TACH_BIN` may identify the exactly matching compiler.

Useful individual commands:

```sh
npm run build --workspace=@tach/browser-test
npm run check --workspace=@tach/browser-test
```

`check` lints and type-checks the standalone harness. `test` rebuilds the
canonical project and performs real browser execution. It writes no custom
report or pass/fail Markdown; process status and assertion diagnostics are the
test contract.

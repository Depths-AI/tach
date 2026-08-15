# Tach

Tach is a small, typed language for portable GPU compute. Its source feels
familiar to a TypeScript developer, while the compiler owns shader entry
points, bindings, byte layout, launch geometry, and target validation.

Write a baseline kernel:

```tach
export function scale[i](
  values: buffer<float32[]>,
  factor: float32,
) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Tach turns it into validated WGSL for WebGPU, validated SPIR-V 1.6 for Vulkan
1.3, typed JavaScript/TypeScript command constructors, reflection metadata,
and readable diagnostics. Tach source contains no WGSL builtins, descriptor
numbers, storage classes, or padding fields.

## Run the first project

Applications need Deno and use npm only to install the package. Install Tach and create one project
manifest at the source root:

```sh
npm install @depths/tach
```

```json
{
  "name": "scaling",
  "version": "0.1.0",
  "javascript": {
    "package": "@example/scaling"
  },
  "docs": {
    "title": "Scaling kernels",
    "summary": "Small GPU transforms used by the application."
  }
}
```

Save the opening kernel as `kernels/scale.tach`. Every Tach source belongs to
exactly one immediate module directory; here the module is `kernels` and the
kernel identity is `kernels/scale`.

Run commands anywhere under the project. Tach walks upward to the nearest
`tach.json`:

```sh
npx tach check
npx tach build
```

`check` validates the complete project through both WebGPU and Vulkan without
writing. `build` atomically replaces `build/` with one cohesive, host-neutral
generated package:

```text
build/
  package.json
  index.js
  index.d.ts
  kernel.wgsl
  kernel.spv
  README.md
  docs/
    kernels.md
```

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/index.js";

const result = await tach(async (gpu) => {
  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
  await gpu.submit(scale(values, 2));
  return values.read();
});

console.log(result); // Float32Array [2, 4, 6, 8]
```

The same import and host model runs in a browser through WebGPU and in Deno
through Tach's Vulkan 1.3 runtime. Host selection is automatic; generated
kernels and application code contain no backend branch. The everyday model has
three operations:

1. `gpu.buffer(value)` creates session-owned GPU state.
2. A generated function such as `scale(...)` constructs a command.
3. `gpu.submit(...)` records and queues commands; `read()` or `idle()` waits
   for completion.

## Source model

Tach has four source forms and three fundamental compiler roles:

| Source form | Tach role | Host exposure |
|---|---|---|
| `function helper(...)` | value helper | none |
| `function stage[i](...)` | private indexed GPU stage | none |
| `export function kernel[i](...)` | indexed stage plus synthesized one-dispatch program | generated JS/TypeScript function |
| `export function program(...)` | explicit orchestration program | generated JS/TypeScript function |

The baseline form shown above is deliberate syntax sugar: an
`export function name[i](...)` declares an indexed stage and a same-named
public one-dispatch program. Simple kernels remain simple.

Struct types are always present in the generated TypeScript API. `export`
applies only to functions and only controls JS/TypeScript exposure; it does
not change visibility between Tach files.

### Projects, modules, and imports

A project contains one or more immediate module directories, each containing
one or more immediate `.tach` kernel files:

```text
simulation/
  tach.json
  data/
    particles.tach
  physics/
    integrate.tach
    step.tach
```

Kernel files import complete kernel files by extensionless project identity:

```tach
import "data/particles";
```

That spelling is literal Tach project identity, never a filesystem-relative,
URL, or npm-package lookup. It contains exactly one `/`; `.tach`, `.`/`..`,
backslashes, extra nesting, and npm-style `@scope/package` names are invalid.

Declarations in the current file and directly imported files are visible.
Imports do not re-export declarations and visibility is not transitive. All
top-level type and function names are project-global and unique, so generated
bindings and both GPU backends share one unambiguous identity for every
declaration.

The compiler rejects cycles in both the kernel-file graph and the graph formed
by collapsing files into modules. Independent declaration work is checked by
bounded workers, then merged in canonical module/kernel/declaration order;
worker scheduling cannot change diagnostics or output bytes.

Coordinates are immutable `uint32` values supplied to GPU invocations, not
host parameters:

```tach
export function image[x, y](pixels: buffer<float32x4[]>) {
  const width = 1920;
  const pixel = y * width + x;
  if (pixel < pixels.length) {
    pixels[pixel] = float32x4(float32(x) / 1920.0, float32(y) / 1080.0, 0.5, 1.0);
  }
}
```

The host supplies a logical size of matching rank:

```ts
await gpu.submit(image(pixels, { size: [1920, 1080] }));
```

Tach rounds each dimension up to whole workgroups. Source must guard any edge
invocations before indexing; the compiler does not add dynamic bounds checks.

Parameters describe the host boundary:

- `buffer<T>` is GPU storage whose read/write mode is inferred from effects;
- a plain type is an immutable value packed by the compiler; and
- `T[]` is a runtime-sized array with `.length` inside indexed stages.

Helpers use TypeScript-shaped declarations and operate only on values:

```tach
type Particle = {
  position: float32x4,
  velocity: float32x4,
};

function advance(particle: Particle, dt: float32): Particle {
  return {
    position: particle.position + particle.velocity * dt,
    velocity: particle.velocity,
  };
}

export function step[i](particles: buffer<Particle[]>, dt: float32) {
  if (i < particles.length) {
    particles[i] = advance(particles[i], dt);
  }
}
```

GPU value types are explicit: `bool`, `int32`, `uint32`, `float32`, and
two-, three-, or four-lane numeric vectors such as `float32x4`. Numeric
literals have no shader suffixes. Context infers their type; conversions use
`int32(value)`, `uint32(value)`, or `float32(value)`.

## Multi-stage programs

When an operation needs several dispatches, keep the indexed work in private
stages and export one orchestration function:

```tach
function multiply[i](
  input: buffer<float32[]>,
  scratch: buffer<float32[]>,
  factor: float32,
) {
  scratch[i] = input[i] * factor;
}

function addBias[i](
  scratch: buffer<float32[]>,
  output: buffer<float32[]>,
  bias: float32,
) {
  output[i] = scratch[i] + bias;
}

export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
  factor: float32,
  bias: float32,
) {
  const scratch = transient<float32>(count);
  run multiply(input, scratch, factor) over count;
  run addBias(scratch, output, bias) over count;
}
```

`transient<T>(length)` requests program-local scratch storage. The compiler
checks its lifetime and type, reuses non-overlapping scratch allocations, and
emits the target execution plan. `run stage(...) over domain` has the same rank
as the stage coordinates. Domains are checked `uint32` shape expressions built
from constants, public values, buffer lengths, arithmetic, `min`, `max`, and
`ceilDiv`.

The public TypeScript API still exposes one command:

```ts
await gpu.submit(transform(input, output, count, 2, 0.5));
```

Tach currently preserves every source `run` as a physical dispatch. It can
move a safe repeated single-stage command into an invocation-local loop, but
it does not fuse distinct stages.

The [language guide](docs/language.md) defines both the baseline sugar and the
complete orchestration language.

## Workgroups and synchronization

The compiler selects portable defaults by stage rank:

| Coordinates | Default workgroup |
|---:|---:|
| `[i]` | `256 x 1 x 1` |
| `[x, y]` | `16 x 16 x 1` |
| `[x, y, z]` | `8 x 8 x 4` |

Most stages should use the default. Specify `@workgroup(...)` when the
algorithm depends on shared memory or barriers:

```tach
@workgroup(64)
export function blockTotals[i](out: buffer<uint32[]>) {
  let partial: shared<uint32[64]>;
  const lane = i % 64;

  partial[lane] = i;
  workgroupBarrier();

  if (lane == 0 && i < out.length) {
    out[i] = partial[0];
  }
}
```

`shared<T>`, atomics, and barriers expose real parallel-memory semantics and
therefore remain explicit. Tach verifies that every workgroup invocation
reaches a barrier under uniform control.

## Structured documentation

`@docs(...)` attaches checked documentation to the kernel file, types, helpers,
stages, and public programs. Names in `field`, `param`, and `coordinate`
clauses are resolved by the compiler, so stale references fail compilation.
Use `//` for single-line implementation notes; block comments are not part of
the language.

```sh
npx tach docs
```

The command atomically refreshes only `build/README.md` and `build/docs/`, so
already compiled target artifacts remain untouched. Every successful build
also refreshes those documents. The root README uses `tach.json` title and
summary, links one document per module, and contains a strict-TypeScript-valid
usage example derived from the same checked public ABI. Module documents place
one section per kernel and cover its internal and exported declarations.

Generated `index.d.ts` carries summaries and parameter context as JSDoc. See the
[documentation section of the language guide](docs/language.md#3-structured-documentation-and-comments).

## Runtime lifetime

`tach(callback)` owns a scoped job: it opens a session, runs the callback,
waits for queued work, returns the callback result, and closes all resources.

Call `tach()` without a callback for resident state:

```ts
import { tach } from "@depths/tach";
import { step } from "./build/index.js";

const gpu = await tach();
const state = gpu.buffer(initialState);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state, 1 / 60));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

`submit()` waits for command preparation and queue submission, not GPU
completion. `idle()`, buffer readback, and scoped-session exit are completion
boundaries. The [TypeScript guide](tach-ts/README.md) defines the full runtime
contract.

## Compiler commands and artifacts

The public `tach` command is delivered by `@depths/tach`. Project commands
operate on the nearest complete project and accept no source-file argument:

```text
tach build [--verbose]
tach check
tach docs
tach fmt
tach instructions [--details <section>...]
tach version
tach help
tach --help
tach -h
```

`tach instructions` works outside a project and prints the dense AI-agent
guide. Its `§N` references are retrievable without emitting the whole reference:
`tach instructions --details 6 7` prints only the import and visibility
sections. Both layers ship inside `@depths/tach`; retrieval needs neither a
Tach project nor the native compiler or network.

Every build emits `index.js`, `index.d.ts`, `kernel.wgsl`, `kernel.spv`,
`package.json`, `README.md`, and one `docs/<module>.md` per module. `--verbose`
adds compiler-owned IR, executable-plan, metadata, and SPIR-V-disassembly files
under `build/diagnostics/`; it does not change executable semantics.

The files form one npm package named by `tach.json.javascript.package`.
`index.js` imports the managed host-neutral implementation from
`@depths/tach/internal`; it neither imports nor re-exports the public `tach`
function. Consumers install `@depths/tach` directly and import the runtime and
generated project separately.

`kernel.wgsl` and the validated SPIR-V 1.6 `kernel.spv` contain the same public
programs and physical stages. The SPIR-V module requires Vulkan 1.3,
Synchronization2, and zero-initialized workgroup memory.

`check` is deliberately targetless: it runs the recoverable frontend, both DAG
checks, unified IR verification/optimization, both backends, binding/package
validation, and documentation rendering entirely in memory. `fmt` formats all
project kernels transactionally using one fixed style. Generated artifacts are
compiler-owned and must never be edited or mixed between runs.

## Compiler architecture

```text
nearest tach.json -> canonical discovery -> import DAGs
                         |
                         v
             concurrent lexing and recovery parsing
                         |
                         v
          global headers + direct-import visibility
                         |
                 bounded semantic work
                         |
                         v
            one merged Kernel IR + Flow IR
                         |
             target-independent optimization
                  /                    \
                 v                      v
       WebGPU executable plan     Vulkan executable plan
         + one WGSL module       + one SPIR-V 1.6 module
                  \                    /
                   v                  v
               bindings, ABI description,
               documentation, validation,
               and atomic build replacement
```

Project loading owns filesystem identity, direct visibility, and canonical
dependency order. Kernel IR owns portable per-invocation meaning. Flow IR owns
dispatch order, resource versions, shapes, and transient lifetimes. Each target then chooses
workgroups, physical entries, parameter blocks, coordinate inputs, and
synchronization without leaking provider concepts upward.

For deeper detail:

- [Language](docs/language.md): complete source syntax and semantic rules.
- [Architecture](docs/architecture.md): ownership and end-to-end data flow.
- [IR](docs/ir.md): Flow IR, Kernel IR, executable plans, and verification.
- [ABI](docs/abi.md): names, layouts, plans, metadata, and host obligations.

## Develop the repository

Compiler development requires Go 1.26.5. TypeScript tooling and every harness
run on Deno. npm is the workspace and package-distribution interface, never the
JavaScript runtime.

```sh
npm ci
npm run check
go test -count=1 ./...
npm test
```

`npm run check` runs Go package-cycle resolution, `go vet`, `staticcheck`,
`deadcode`, duplicate analysis, all Go tests, the native bridge build, package
generation, Deno lint/type checking, and instruction-bundle validation. The
committed Go tool directive pins
`dupl`; no separate installation is required. `npm test` then executes the
package unit suite, Chromium/WebGPU corpus, Deno/Vulkan corpus, and dual-host
showcase.

The ordinary Go suite always runs the seed corpus for the recoverable lexer,
parser, semantic checker, and formatter invariants. Exercise mutation-driven
fuzzing explicitly when changing those seams:

```sh
go test ./src/lexer -run '^$' -fuzz FuzzLexerRecoverySpansStayBounded
go test ./src/parser -run '^$' -fuzz FuzzParserRecoveryIsTotalAndDeterministic
go test ./src/sema -run '^$' -fuzz FuzzSemanticCheckingReturnsSourceDiagnostics
go test ./src/compiler -run '^$' -fuzz FuzzFormatterPreservesSyntaxAndStabilizes
```

These properties require bounded spans and guaranteed lexer termination,
deterministic partial ASTs and diagnostics, source-facing semantic failures,
and formatting that preserves the normalized token/comment/string stream,
parses again, and reaches an idempotent fixed point.

Install Chrome or Chromium and run every real hardware harness:

```sh
npm test
```

Run just the native path through Khronos validation and Vulkan:

```sh
npm test --workspace=@tach/deno-test
```

That harness additionally needs CGO, a C compiler, a Vulkan loader and driver,
Khronos `spirv-val`, and preferably the Vulkan validation layer. Harness details
live in [browser-test](browser-test/README.md),
[deno-test](deno-test/README.md), and
[showcase-ts](showcase-ts/README.md).

## Repository map

```text
main.go          private native compiler-engine operations
src/lexer        tokens and source spelling
src/parser       grammar and source-shaped AST construction
src/ast          source representation
src/types        logical type system
src/sema         checking plus Kernel IR and Flow IR lowering
src/ir           per-invocation Kernel IR
src/flow         public-program Flow IR
src/opt          target-independent Kernel IR optimization
src/backend      target executable planning and Vulkan/SPIR-V profile
src/layout       canonical host-visible byte layout
src/abi          private names and parameter blocks
src/wgsl         WGSL lowering, emission, and validation
src/spirv        SPIR-V lowering, emission, decoding, and validation
src/bindings     runtime metadata and target-neutral project descriptions
src/compiler     project discovery, DAGs, formatting, orchestration, and native staging
tach-ts          Deno-first compiler delivery plus WebGPU and Vulkan runtime
examples         maintained multi-module Tach project and execution corpus
browser-test     Chromium WebGPU correctness harness
deno-test        Deno/Vulkan correctness harness over the same examples
native           Tach-owned Vulkan 1.3 FFI runtime
showcase-ts      six-workload dual-host large GPU benchmark harness
```

## Releases and license

Maintainers can run all release checks, build six native compiler targets,
pack the npm package, and write checksums with:

```sh
./release.sh v0.1.0
```

Adding `--publish` creates the GitHub release and publishes npm; it requires
authenticated GitHub and npm CLIs plus a clean committed tree.

Tach is licensed under the [GNU Affero General Public License version 3](LICENSE)
only (`AGPL-3.0-only`).

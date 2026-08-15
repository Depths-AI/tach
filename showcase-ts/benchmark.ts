import { tach, type TachAdapterInfo } from "@depths/tach";
import {
  type BenchmarkReport,
  type BenchmarkRun,
  runBenchmarks,
} from "./src/benchmarks.ts";

interface HostRun extends BenchmarkRun {
  readonly adapter: TachAdapterInfo;
}
interface PublishedHost {
  readonly adapter: TachAdapterInfo;
  readonly report: BenchmarkReport;
}

const root = new URL("./", import.meta.url);
const vulkan = await tach(
  async (gpu): Promise<HostRun> => ({
    adapter: gpu.adapter,
    ...await runBenchmarks(gpu),
  }),
  { powerPreference: "high-performance" },
);

function chromePath(): string {
  const programFiles = Deno.env.get("ProgramFiles"),
    local = Deno.env.get("LOCALAPPDATA");
  const candidates = Deno.build.os === "windows"
    ? [
      Deno.env.get("CHROME_BIN"),
      programFiles &&
      `${programFiles}\\Google\\Chrome\\Application\\chrome.exe`,
      local && `${local}\\Google\\Chrome\\Application\\chrome.exe`,
    ]
    : Deno.build.os === "darwin"
    ? [
      Deno.env.get("CHROME_BIN"),
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    : [
      Deno.env.get("CHROME_BIN"),
      "/usr/bin/google-chrome",
      "/usr/bin/chromium",
      "/usr/bin/chromium-browser",
    ];
  for (const candidate of candidates) {
    if (candidate) {
      try {
        if (Deno.statSync(candidate).isFile) return candidate;
      } catch { /* Try the next platform path. */ }
    }
  }
  throw new Error("Chrome is unavailable; set CHROME_BIN");
}

class Protocol {
  #id = 0;
  #pending = new Map<
    number,
    { resolve(value: unknown): void; reject(error: unknown): void }
  >();
  constructor(readonly socket: WebSocket) {
    socket.onmessage = ({ data }) => {
      const message = JSON.parse(String(data));
      if (!message.id) return;
      const pending = this.#pending.get(message.id);
      if (!pending) return;
      this.#pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message));
      else pending.resolve(message.result);
    };
    socket.onclose = () => {
      for (const pending of this.#pending.values()) {
        pending.reject(new Error("Chrome closed"));
      }
      this.#pending.clear();
    };
  }
  call<T>(method: string, params: object = {}): Promise<T> {
    const id = ++this.#id;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
      });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }
}

async function webGPU(): Promise<HostRun> {
  let posted: PublishedHost | undefined;
  const frames: Record<string, Uint32Array> = {}, abort = new AbortController();
  const server = Deno.serve({
    hostname: "127.0.0.1",
    port: 0,
    signal: abort.signal,
    onListen() {},
  }, async (request) => {
    const path = new URL(request.url).pathname;
    if (path === "/") {
      return new Response('<script type="module" src="/app.js"></script>', {
        headers: { "content-type": "text/html" },
      });
    }
    if (path === "/app.js" || path === "/kernel.wgsl") {
      return new Response(await Deno.readFile(new URL(`dist${path}`, root)), {
        headers: {
          "content-type": path.endsWith(".js")
            ? "text/javascript"
            : "text/plain",
        },
      });
    }
    if (request.method === "POST" && path === "/result") {
      posted = await request.json() as PublishedHost;
      return new Response(null, { status: 204 });
    }
    if (request.method === "POST" && path.startsWith("/frame/")) {
      frames[path.slice(7)] = new Uint32Array(await request.arrayBuffer());
      return new Response(null, { status: 204 });
    }
    return new Response("not found", { status: 404 });
  });
  const probe = Deno.listen({ hostname: "127.0.0.1", port: 0 }),
    debuggingPort = (probe.addr as Deno.NetAddr).port;
  probe.close();
  const profile = await Deno.makeTempDir({ prefix: "tach-showcase-" });
  const flags = [
    `--remote-debugging-port=${debuggingPort}`,
    `--user-data-dir=${profile}`,
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--enable-gpu",
    "--enable-unsafe-webgpu",
    "--enable-unsafe-swiftshader",
    "--remote-allow-origins=*",
  ];
  if (Deno.build.os === "linux") {
    flags.push(
      "--use-angle=vulkan",
      "--enable-features=Vulkan",
      "--disable-vulkan-surface",
    );
  }
  const chrome = new Deno.Command(chromePath(), {
    args: flags,
    stdin: "null",
    stdout: "null",
    stderr: "null",
  }).spawn();
  try {
    const endpoint = `http://127.0.0.1:${debuggingPort}`;
    for (let attempt = 0;; attempt++) {
      try {
        if ((await fetch(`${endpoint}/json/version`)).ok) break;
      } catch { /* Chrome may still be starting. */ }
      if (attempt === 199) {
        throw new Error("Chrome debugging endpoint did not start");
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    const page = await fetch(
      `${endpoint}/json/new?${
        encodeURIComponent(
          `http://127.0.0.1:${(server.addr as Deno.NetAddr).port}/`,
        )
      }`,
      { method: "PUT" },
    ).then((response) => response.json());
    const socket = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise<void>((resolve, reject) => {
      socket.onopen = () => resolve();
      socket.onerror = () =>
        reject(new Error("Chrome protocol connection failed"));
    });
    const protocol = new Protocol(socket);
    await protocol.call("Runtime.enable");
    for (let attempt = 0;; attempt++) {
      const evaluation = await protocol.call<
        {
          readonly exceptionDetails?: {
            readonly text: string;
            readonly exception?: { readonly description?: string };
          };
          readonly result: { readonly value?: unknown };
        }
      >("Runtime.evaluate", {
        expression:
          "globalThis.__tachShowcase ? globalThis.__tachShowcase : null",
        awaitPromise: true,
        returnByValue: true,
      });
      if (evaluation.exceptionDetails) {
        throw new Error(
          evaluation.exceptionDetails.exception?.description ??
            evaluation.exceptionDetails.text,
        );
      }
      if (evaluation.result.value) break;
      if (attempt === 1_199) {
        throw new Error("browser showcase did not complete");
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    socket.close();
    if (!posted) throw new Error("browser showcase did not publish results");
    return { ...posted, frames };
  } finally {
    chrome.kill();
    await chrome.status;
    abort.abort();
    await Deno.remove(profile, { recursive: true });
  }
}

const web = await webGPU(), generatedAt = new Date().toISOString();
const hosts: readonly HostRun[] = [web, vulkan];
const published = {
  generatedAt,
  hosts: hosts.map(({ adapter, report }) => ({ adapter, report })),
};

const cell = (value: unknown): string => String(value).replaceAll("|", "\\|");
const number = (value: string | number | boolean): string =>
  typeof value === "number"
    ? Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(4)
    : String(value);
function markdown(): string {
  const lines = [
    "# Tach large GPU showcase",
    "",
    `Generated: ${generatedAt}`,
    "",
    "## Results",
    "",
    "| Backend | Adapter | Category | Workload | Median | Raw samples (ms) | Throughput | FPS | Validation |",
    "|---|---|---|---|---:|---|---:|---:|---|",
  ];
  for (const host of hosts) {
    for (const result of host.report.results) {
      lines.push(
        `| ${host.adapter.backend} | ${
          cell(host.adapter.name)
        } | ${result.category} | ${cell(result.name)} | ${
          result.gpuMs.toFixed(3)
        } ms | ${
          result.gpuSamplesMs.map((sample) => sample.toFixed(3)).join(", ")
        } | ${result.throughput.toFixed(3)} ${result.throughputUnit} | ${
          result.framesPerSecond?.toFixed(1) ?? "-"
        } | ${cell(result.check)} |`,
      );
    }
  }
  for (const host of hosts) {
    lines.push(
      "",
      `## ${host.adapter.backend}: ${host.adapter.name}`,
      "",
      `Timing: ${host.report.timing}`,
      "",
    );
    for (const result of host.report.results) {
      lines.push(
        `### ${result.name}`,
        "",
        result.problem,
        "",
        `Dispatches per timed sample: ${result.dispatches}`,
        "",
        "| Detail | Value |",
        "|---|---:|",
      );
      for (const [key, value] of Object.entries(result.details)) {
        lines.push(`| ${cell(key)} | ${cell(number(value))} |`);
      }
      lines.push("");
    }
  }
  lines.push(
    "## Contract",
    "",
    "Both hosts execute the same six Tach programs and generated package facade. Each workload receives one untimed warmup followed by five timed samples in one persistent Tach session. Every sample measures command submission through GPU completion. Allocation, initial upload, readback, PNG encoding, report generation, and validation are excluded.",
    "",
  );
  return lines.join("\n");
}

const crcTable = Uint32Array.from({ length: 256 }, (_, byte) => {
  let value = byte;
  for (let bit = 0; bit < 8; bit++) {
    value = value & 1 ? 0xedb8_8320 ^ value >>> 1 : value >>> 1;
  }
  return value >>> 0;
});
function chunk(name: string, data: Uint8Array): Uint8Array {
  const type = new TextEncoder().encode(name),
    output = new Uint8Array(data.length + 12),
    view = new DataView(output.buffer);
  view.setUint32(0, data.length);
  output.set(type, 4);
  output.set(data, 8);
  let crc = 0xffff_ffff;
  for (const byte of output.subarray(4, data.length + 8)) {
    crc = crcTable[(crc ^ byte) & 255]! ^ crc >>> 8;
  }
  view.setUint32(data.length + 8, (crc ^ 0xffff_ffff) >>> 0);
  return output;
}
async function png(pixels: Uint32Array): Promise<Uint8Array> {
  const width = 1920,
    height = 1080,
    row = width * 4,
    raw = new Uint8Array((row + 1) * height),
    bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  for (let y = 0; y < height; y++) {
    raw.set(bytes.subarray(y * row, (y + 1) * row), y * (row + 1) + 1);
  }
  const compressed = new Uint8Array(
    await new Response(
      new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate")),
    ).arrayBuffer(),
  );
  const header = new Uint8Array(13), view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header.set([8, 6, 0, 0, 0], 8);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header),
    chunk("IDAT", compressed),
    chunk("IEND", new Uint8Array()),
  ];
  const output = new Uint8Array(
    parts.reduce((sum, part) => sum + part.length, 0),
  );
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

const reports = new URL("reports/", root);
await Deno.remove(reports, { recursive: true }).catch((error) => {
  if (!(error instanceof Deno.errors.NotFound)) throw error;
});
await Deno.mkdir(reports);
await Promise.all([
  Deno.writeTextFile(
    new URL("reports/gpu.json", root),
    `${JSON.stringify(published, null, 2)}\n`,
  ),
  Deno.writeTextFile(new URL("reports/gpu.md", root), markdown()),
  ...hosts.flatMap((host) =>
    Object.entries(host.frames).map(async ([name, frame]) =>
      Deno.writeFile(
        new URL(`reports/${host.adapter.backend}-${name}.png`, root),
        await png(frame),
      )
    )
  ),
]);
for (const host of hosts) {
  console.log(`${host.adapter.backend}: ${host.adapter.name}`);
  console.table(host.report.results.map((result) => ({
    category: result.category,
    workload: result.name,
    medianMs: result.gpuMs.toFixed(2),
    throughput: `${result.throughput.toFixed(2)} ${result.throughputUnit}`,
    fps: result.framesPerSecond?.toFixed(1) ?? "-",
  })));
}

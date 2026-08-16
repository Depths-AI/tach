const root = new URL("./", import.meta.url), abort = new AbortController();
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
  if (path !== "/app.js" && path !== "/kernel.wgsl.gz") {
    return new Response("not found", { status: 404 });
  }
  return new Response(await Deno.readFile(new URL(`dist${path}`, root)), {
    headers: {
      "content-type": path.endsWith(".js")
        ? "text/javascript"
        : "application/gzip",
    },
  });
});

function browser(): string {
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

const probe = Deno.listen({ hostname: "127.0.0.1", port: 0 }),
  debuggingPort = (probe.addr as Deno.NetAddr).port;
probe.close();
const profile = await Deno.makeTempDir({ prefix: "tach-chrome-" });
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
const chrome = new Deno.Command(browser(), {
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
        readonly result: {
          readonly value?: {
            readonly adapter: { readonly name: string };
            readonly programs: number;
          };
        };
      }
    >("Runtime.evaluate", {
      expression: "globalThis.__tachTest ? globalThis.__tachTest : null",
      awaitPromise: true,
      returnByValue: true,
    });
    if (evaluation.exceptionDetails) {
      throw new Error(
        evaluation.exceptionDetails.exception?.description ??
          evaluation.exceptionDetails.text,
      );
    }
    if (evaluation.result.value) {
      console.log(
        `WebGPU execution: ${evaluation.result.value.adapter.name}; ${evaluation.result.value.programs} programs`,
      );
      break;
    }
    if (attempt === 199) throw new Error("browser test did not complete");
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  socket.close();
} finally {
  chrome.kill();
  await chrome.status;
  abort.abort();
  await Deno.remove(profile, { recursive: true });
}

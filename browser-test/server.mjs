import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const host = "127.0.0.1";
const port = Number.parseInt(process.env.PORT ?? "4173", 10);
const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".spv", "application/octet-stream"],
  [".tach", "text/plain; charset=utf-8"],
  [".tir", "text/plain; charset=utf-8"],
  [".wgsl", "text/plain; charset=utf-8"],
]);

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? "/", `http://${host}:${port}`);
  if (url.pathname === "/health") {
    response.writeHead(200, { "content-type": "text/plain; charset=utf-8" });
    response.end("ok\n");
    return;
  }

  let pathname;
  try {
    pathname = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
  } catch {
    response.writeHead(400);
    response.end("bad request\n");
    return;
  }

  const filePath = resolve(root, `.${pathname}`);
  if (filePath !== root && !filePath.startsWith(root + sep)) {
    response.writeHead(403);
    response.end("forbidden\n");
    return;
  }

  try {
    const info = await stat(filePath);
    if (!info.isFile()) throw new Error("not a file");
    response.writeHead(200, {
      "cache-control": "no-store",
      "content-length": info.size,
      "content-type": contentTypes.get(extname(filePath)) ?? "application/octet-stream",
      "cross-origin-embedder-policy": "require-corp",
      "cross-origin-opener-policy": "same-origin",
    });
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("not found\n");
  }
});

server.listen(port, host, () => {
  console.log(`Tach browser harness listening on http://${host}:${port}`);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}

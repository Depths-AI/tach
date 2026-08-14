import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const examplesBuild = join(root, "..", "examples", "build");
const tachDist = dirname(fileURLToPath(import.meta.resolve("@depths/tach")));
const tachPrefix = "/@depths/tach/";
const host = "127.0.0.1";
const port = Number.parseInt(process.env.PORT ?? "4173", 10);
const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
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

  const generated = pathname.startsWith("/build/");
  const contentRoot = pathname.startsWith(tachPrefix) ? tachDist : generated ? examplesBuild : root;
  const contentPath = pathname.startsWith(tachPrefix)
    ? pathname.slice(tachPrefix.length)
    : generated ? pathname.slice("/build/".length) : pathname.slice(1);
  const filePath = resolve(contentRoot, contentPath);
  if (filePath !== contentRoot && !filePath.startsWith(contentRoot + sep)) {
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

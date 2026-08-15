import { build } from "@depths/tach/compiler";

const here = new URL("../", import.meta.url),
  source = new URL("../examples/", here),
  destination = new URL("generated/", here);

async function copy(from: URL, to: URL): Promise<void> {
  await Deno.mkdir(to, { recursive: true });
  for await (const entry of Deno.readDir(from)) {
    const source = new URL(
        `${entry.name}${entry.isDirectory ? "/" : ""}`,
        from,
      ),
      target = new URL(`${entry.name}${entry.isDirectory ? "/" : ""}`, to);
    if (entry.isDirectory) await copy(source, target);
    else await Deno.copyFile(source, target);
  }
}

await build({
  cwd: decodeURIComponent(source.pathname).replace(/^\/(.:\/)/u, "$1"),
});
await Deno.remove(destination, { recursive: true }).catch((error) => {
  if (!(error instanceof Deno.errors.NotFound)) throw error;
});
await copy(new URL("build/", source), destination);
await Deno.remove(new URL("dist/", here), { recursive: true }).catch(
  (error) => {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
  },
);
const bundled = await new Deno.Command(Deno.execPath(), {
  args: [
    "bundle",
    "--quiet",
    "--platform",
    "browser",
    "--output",
    "dist/app.js",
    "src/main.ts",
  ],
  cwd: here,
  stdin: "null",
  stdout: "inherit",
  stderr: "inherit",
}).output();
if (!bundled.success) throw new Error(`deno bundle exited ${bundled.code}`);
await Deno.copyFile(
  new URL("generated/kernel.wgsl", here),
  new URL("dist/kernel.wgsl", here),
);

import { build } from "@depths/tach/compiler";

const root = new URL("../", import.meta.url);
await build({
  cwd: decodeURIComponent(root.pathname).replace(/^\/(.:\/)/u, "$1"),
});
await Deno.remove(new URL("dist/", root), { recursive: true }).catch(
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
    "src/browser.ts",
  ],
  cwd: root,
  stdin: "null",
  stdout: "inherit",
  stderr: "inherit",
}).output();
if (!bundled.success) throw new Error(`deno bundle exited ${bundled.code}`);
await Deno.copyFile(
  new URL("build/kernel.wgsl", root),
  new URL("dist/kernel.wgsl", root),
);

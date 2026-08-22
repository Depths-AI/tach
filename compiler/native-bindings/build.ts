const names: Readonly<Record<string, string>> = {
  "linux:x86_64": "tach-vulkan.linux.x86_64.so",
  "windows:x86_64": "tach-vulkan.windows.x86_64.dll",
};
const name = names[`${Deno.build.os}:${Deno.build.arch}`];
if (!name) {
  throw new Error(
    `unsupported native Tach host ${Deno.build.os}/${Deno.build.arch}`,
  );
}
const root = new URL("../../", import.meta.url),
  sdk = Deno.env.get("VULKAN_SDK");
if (Deno.build.os === "windows" && !sdk) {
  throw new Error("VULKAN_SDK must name an official Vulkan SDK");
}
const compiler = await new Deno.Command("go", {
  args: ["env", "CC"],
  stdout: "piped",
}).output();
if (!compiler.success) throw new Error(`go env CC exited ${compiler.code}`);
const ccheck = await new Deno.Command(
  new TextDecoder().decode(compiler.stdout).trim(),
  {
    args: [
      "-std=c11",
      "-Wall",
      "-Wextra",
      "-Werror",
      ...(Deno.build.os === "windows" ? ["-Wno-cast-function-type"] : []),
      "-fsyntax-only",
      ...(sdk ? [`-I${sdk.replaceAll("\\", "/")}/Include`] : []),
      "compiler/native-bindings/vulkan.c",
    ],
    cwd: root,
    stdin: "null",
    stdout: "inherit",
    stderr: "inherit",
  },
).output();
if (!ccheck.success) throw new Error(`native C check exited ${ccheck.code}`);
await Deno.mkdir(new URL("tach/native/", root), { recursive: true });
for (
  const args of [
    ["test", "-race", "-tags", "tachvulkan", "./native-bindings"],
    [
      "build",
      "-tags",
      "tachvulkan",
      "-buildmode=c-shared",
      "-trimpath",
      "-o",
      `../tach/native/${name}`,
      "./native-bindings",
    ],
  ]
) {
  const result = await new Deno.Command("go", {
    args,
    cwd: new URL("../", import.meta.url),
    env: {
      CGO_ENABLED: "1",
      ...(sdk ? { CGO_CFLAGS: `-I${sdk.replaceAll("\\", "/")}/Include` } : {}),
    },
    stdin: "null",
    stdout: "inherit",
    stderr: "inherit",
  }).output();
  if (!result.success) throw new Error(`go ${args[0]} exited ${result.code}`);
}
await Deno.remove(
  new URL(`tach/native/${name.replace(/\.[^.]+$/u, ".h")}`, root),
).catch((error) => {
  if (!(error instanceof Deno.errors.NotFound)) throw error;
});

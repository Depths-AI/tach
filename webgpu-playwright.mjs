// Prefer hardware; permit Chromium's SwiftShader fallback on GPU-less hosts.
const args = [
  "--enable-gpu",
  "--enable-unsafe-webgpu",
  "--enable-unsafe-swiftshader",
];

if (process.platform === "linux") {
  args.push("--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface");
}

export const webGPU = {
  channel: "chromium",
  headless: true,
  launchOptions: { args },
};

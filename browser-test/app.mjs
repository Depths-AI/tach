const gpuStatus = document.querySelector("#gpu-status");
const buildStatus = document.querySelector("#build-status");
const examplesBody = document.querySelector("#examples");
const probeButton = document.querySelector("#probe");

function adapterReport(info) {
  const identity = info.description || [info.vendor, info.architecture].filter(Boolean).join(" ") || "unnamed adapter";
  const identityLooksSoftware = /swiftshader|software|llvmpipe|lavapipe|softpipe|warp|basic render/i.test(
    [info.description, info.vendor, info.architecture, info.device].filter(Boolean).join(" "),
  );
  const software = info.isFallbackAdapter === true || identityLooksSoftware;
  return {
    identity,
    mode: software ? "software-emulated" : "hardware-accelerated",
    software,
  };
}

async function probeWebGPU() {
  gpuStatus.className = "";
  gpuStatus.textContent = "probing…";
  if (!("gpu" in navigator)) {
    gpuStatus.className = "warn";
    gpuStatus.textContent = "API unavailable";
    return { available: false, reason: "navigator.gpu is unavailable" };
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    gpuStatus.className = "warn";
    gpuStatus.textContent = "API present, no adapter";
    return { available: false, reason: "no WebGPU adapter" };
  }
  const info = adapter.info ?? {};
  const report = adapterReport(info);
  gpuStatus.className = "ok";
  gpuStatus.textContent = `${report.mode}: ${report.identity}`;
  return {
    available: true,
    info: {
      architecture: info.architecture ?? "",
      description: info.description ?? "",
      device: info.device ?? "",
      isFallbackAdapter: info.isFallbackAdapter ?? null,
      vendor: info.vendor ?? "",
    },
    ...report,
  };
}

async function loadExamples() {
  const response = await fetch("/build/manifest.json");
  if (!response.ok) throw new Error("run npm run build:examples before starting the harness");
  const manifest = await response.json();
  examplesBody.replaceChildren();
  for (const entry of manifest.examples) {
    const row = document.createElement("tr");
    const cells = [
      entry.name,
      entry.kernels.join(", "),
      String(entry.resources),
      `${entry.wgslBytes} bytes`,
    ];
    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    examplesBody.append(row);
  }
  const version = document.createElement("code");
  version.textContent = manifest.cliVersion;
  buildStatus.replaceChildren(version, ` generated ${manifest.examples.length} modules`);
  return manifest;
}

probeButton.addEventListener("click", () => void probeWebGPU());
globalThis.__tachHarnessReady = Promise.all([loadExamples(), probeWebGPU()]);

import { tach } from "@depths/tach";
import { integrate, type Particle } from "../build/particles.js";
import "./style.css";

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`showcase is missing ${selector}`);
  return element;
}

const status = required<HTMLParagraphElement>("#status");
const output = required<HTMLPreElement>("#output");

async function main(): Promise<void> {
  const initial: readonly Particle[] = [
    {
      position: [1, 2, 3, 1],
      velocity: [2, 4, 6, 0],
    },
    {
      position: [-1, -2, -3, 1],
      velocity: [1, 2, 3, 0],
    },
  ];
  const result = await tach(async (gpu) => {
    const particles = gpu.buffer(initial);
    await integrate(particles, {
      dt: 0.5,
      count: initial.length,
    });
    return particles.read();
  });

  if (!result.ok) {
    status.textContent = "The showcase could not run.";
    output.textContent = `[${result.error.code}] ${result.error.message}`;
    return;
  }

  status.textContent = `Integrated ${result.value.length} particles on WebGPU.`;
  output.textContent = JSON.stringify(result.value, null, 2);
}

void main().catch((error: unknown) => {
  status.textContent = "The showcase could not run.";
  output.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
});

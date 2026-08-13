import { spawnSync } from "node:child_process";

const result = spawnSync("go", ["tool", "dupl", "-plumbing", "-t", "75", "src"], { encoding: "utf8" });
if (result.error) throw result.error;
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status ?? 1);
if (result.stdout.trim()) {
  console.error("dupl: remove the reported clones or raise the threshold only with explicit review");
  process.exit(1);
}

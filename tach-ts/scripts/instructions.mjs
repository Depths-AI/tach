import { readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
if (args.length > 1 || (args[0] && args[0] !== "--check")) {
  throw new Error("usage: node scripts/instructions.mjs [--check]");
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const [miniSource, fullSource] = await Promise.all([
  readFile(join(root, "docs", "INSTRUCTION_MINI.md"), "utf8"),
  readFile(join(root, "docs", "INSTRUCTIONS.md"), "utf8"),
]);
const mini = miniSource.replaceAll("\r\n", "\n").trim();
const full = fullSource.replaceAll("\r\n", "\n").trim();
const headings = [...full.matchAll(/^## ([1-9]\d*)\. ([^\n]+)$/gmu)];
if (!headings.length || headings.length !== [...full.matchAll(/^## /gmu)].length) {
  throw new Error("every level-two instruction heading must be numbered");
}

const sections = {};
for (const [index, heading] of headings.entries()) {
  const number = index + 1;
  if (Number(heading[1]) !== number) throw new Error(`expected instruction section ${number}`);
  const start = heading.index + heading[0].length;
  const markdown = full.slice(start, headings[index + 1]?.index ?? full.length).trim();
  if (!markdown) throw new Error(`instruction section ${number} is empty`);
  sections[number] = { title: heading[2], markdown };
}

const words = mini.match(/\S+/gu)?.length ?? 0;
if (words < 1_500 || words > 2_000) throw new Error(`mini instructions contain ${words} words; expected 1500-2000`);
const references = new Set([...mini.matchAll(/§([1-9]\d*)/gu)].map((match) => Number(match[1])));
const missing = headings.map((_, index) => index + 1).filter((number) => !references.has(number));
if (missing.length) throw new Error(`mini instructions do not reference sections ${missing.join(", ")}`);

if (args[0] !== "--check") {
  const output = join(root, "tach-ts", "dist", "instructions.json");
  await writeFile(output, `${JSON.stringify({ schema: 1, mini, sections })}\n`, "utf8");
}

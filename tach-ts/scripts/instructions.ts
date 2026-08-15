const args = Deno.args;
if (args.length > 1 || args[0] && args[0] !== "--check") {
  throw new Error("usage: deno run scripts/instructions.ts [--check]");
}

const root = new URL("../../", import.meta.url);
const [miniSource, fullSource] = await Promise.all([
  Deno.readTextFile(new URL("docs/INSTRUCTION_MINI.md", root)),
  Deno.readTextFile(new URL("docs/INSTRUCTIONS.md", root)),
]);
const mini = miniSource.replaceAll("\r\n", "\n").trim(),
  full = fullSource.replaceAll("\r\n", "\n").trim();
const headings = [...full.matchAll(/^## ([1-9]\d*)\. ([^\n]+)$/gmu)];
if (
  !headings.length || headings.length !== [...full.matchAll(/^## /gmu)].length
) throw new Error("every level-two instruction heading must be numbered");

const sections: Record<
  string,
  { readonly title: string; readonly markdown: string }
> = {};
for (const [index, heading] of headings.entries()) {
  const number = index + 1;
  if (Number(heading[1]) !== number) {
    throw new Error(`expected instruction section ${number}`);
  }
  const start = heading.index + heading[0].length,
    markdown = full.slice(start, headings[index + 1]?.index ?? full.length)
      .trim();
  if (!markdown) throw new Error(`instruction section ${number} is empty`);
  sections[number] = { title: heading[2]!, markdown };
}

const words = mini.match(/\S+/gu)?.length ?? 0;
if (words < 1_500 || words > 2_000) {
  throw new Error(
    `mini instructions contain ${words} words; expected 1500-2000`,
  );
}
const references = new Set(
  [...mini.matchAll(/§([1-9]\d*)/gu)].map((match) => Number(match[1])),
);
const missing = headings.map((_, index) => index + 1).filter((number) =>
  !references.has(number)
);
if (missing.length) {
  throw new Error(
    `mini instructions do not reference sections ${missing.join(", ")}`,
  );
}

if (args[0] !== "--check") {
  await Deno.writeTextFile(
    new URL("tach-ts/dist/instructions.json", root),
    `${JSON.stringify({ schema: 1, mini, sections })}\n`,
  );
}

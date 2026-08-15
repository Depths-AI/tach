const result = await new Deno.Command("go", {
  args: ["tool", "dupl", "-plumbing", "-t", "75", "src", "native", "main.go"],
  stdout: "piped",
  stderr: "inherit",
}).output();
await Deno.stdout.write(result.stdout);
if (!result.success) Deno.exit(result.code);
if (new TextDecoder().decode(result.stdout).trim()) {
  console.error(
    "dupl: remove the reported clones or raise the threshold only with explicit review",
  );
  Deno.exit(1);
}

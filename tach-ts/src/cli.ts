#!/usr/bin/env -S deno run --allow-read --allow-write --allow-env --allow-run --allow-net

import {
  build,
  check,
  CompilerError,
  docs,
  format,
  packageVersion,
  type ProjectResult,
  renderDiagnostics,
} from "./compiler.ts";

const help = `Tach - lean typed GPGPU compiler

Usage:
  tach build [--verbose] [--json]
  tach check [--json]
  tach docs [--json]
  tach fmt [--json]
  tach instructions [--details <section>...]
  tach version
  tach help

Commands:
  build         build the complete browser and Deno/Vulkan package
  check         validate the complete WebGPU and Vulkan project pipeline without writing
  docs          refresh only generated project documentation
  fmt           format every kernel in the nearest Tach project
  instructions  print AI-agent guidance or selected numbered detail sections
  version       print the installed Tach version
  help          print this help

Options:
  --json        emit one machine-readable result instead of human prose
`;

function write(value: string): Promise<void> {
  return Deno.stdout.write(new TextEncoder().encode(value)).then(() => {});
}

const args = Deno.args;
const json = args.includes("--json");

function flags(allowed: readonly string[]): Set<string> {
  const values = new Set(args.slice(1));
  if (
    values.size !== args.length - 1 ||
    [...values].some((value) => !allowed.includes(value))
  ) {
    throw new Error(
      `usage: tach ${args[0]} ${allowed.map((flag) => `[${flag}]`).join(" ")}`
        .trimEnd(),
    );
  }
  return values;
}

async function complete(
  command: string,
  message: string,
  result?: ProjectResult,
): Promise<void> {
  if (json) {
    await write(
      `${
        JSON.stringify({
          schema: 1,
          ok: true,
          command,
          root: result?.root,
          diagnostics: result?.diagnostics ?? [],
        })
      }\n`,
    );
  } else {
    if (result?.diagnostics.length) {
      console.error(renderDiagnostics(result.diagnostics));
    }
    console.log(message);
  }
}

try {
  switch (args[0]) {
    case "help":
    case "-h":
    case "--help":
      if (args.length !== 1) throw new Error(`${args[0]} accepts no arguments`);
      await write(help);
      break;
    case "version":
      if (args.length !== 1) throw new Error("version accepts no arguments");
      console.log(`tach ${await packageVersion()}`);
      break;
    case "instructions": {
      if (args.length !== 1 && (args[1] !== "--details" || args.length < 3)) {
        throw new Error("usage: tach instructions [--details <section>...]");
      }
      const bundle = JSON.parse(
        await Deno.readTextFile(
          new URL("../dist/instructions.json", import.meta.url),
        ),
      ) as {
        readonly mini: string;
        readonly sections: Readonly<
          Record<string, { readonly title: string; readonly markdown: string }>
        >;
      };
      if (args.length === 1) {
        await write(`${bundle.mini}\n`);
        break;
      }
      const selected = [...new Set(args.slice(2))].map((number) => {
        const section = bundle.sections[number];
        if (!/^[1-9]\d*$/u.test(number) || !section) {
          throw new Error(
            `instruction section ${JSON.stringify(number)} does not exist`,
          );
        }
        return `## ${number}. ${section.title}\n\n${section.markdown}`;
      });
      await write(`${selected.join("\n\n")}\n`);
      break;
    }
    case "build": {
      const options = flags(["--verbose", "--json"]);
      const result = await build({ verbose: options.has("--verbose") });
      await complete("build", `built ${result.root}`, result);
      break;
    }
    case "check":
      flags(["--json"]);
      await complete("check", "ok", await check());
      break;
    case "docs":
      flags(["--json"]);
      await complete("docs", "updated documentation", await docs());
      break;
    case "fmt":
      flags(["--json"]);
      await format();
      await complete("fmt", "formatted");
      break;
    default:
      throw new Error(
        args.length === 0
          ? "a command is required; run `tach help`"
          : `unknown command ${JSON.stringify(args[0])}`,
      );
  }
} catch (error) {
  const value = error as {
    readonly code?: unknown;
    readonly message?: unknown;
  };
  const message = typeof value.message === "string"
    ? value.message
    : String(error);
  if (json) {
    await write(`${
      JSON.stringify({
        schema: 1,
        ok: false,
        command: args[0] ?? null,
        diagnostics: error instanceof CompilerError ? error.diagnostics : [],
        ...(error instanceof CompilerError ? {} : {
          error: {
            code: typeof value.code === "string" ? value.code : "usage",
            message,
          },
        }),
      })
    }\n`);
  } else {
    console.error(
      error instanceof CompilerError
        ? message
        : `tach: ${
          typeof value.code === "string" ? `[${value.code}] ` : ""
        }${message}`,
    );
  }
  Deno.exitCode = 1;
}

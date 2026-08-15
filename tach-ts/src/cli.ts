#!/usr/bin/env -S deno run --allow-read --allow-write --allow-env --allow-run --allow-net

import { build, check, docs, format, packageVersion } from "./compiler.ts";

const help = `Tach — lean typed GPGPU compiler

Usage:
  tach build [--verbose]
  tach check
  tach docs
  tach fmt
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
`;

function write(value: string): Promise<void> {
  return Deno.stdout.write(new TextEncoder().encode(value)).then(() => {});
}

const args = Deno.args;
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
      if (args.length > 2 || args.length === 2 && args[1] !== "--verbose") {
        throw new Error("usage: tach build [--verbose]");
      }
      const result = await build({ verbose: args.length === 2 });
      console.log(`built ${result.root}`);
      break;
    }
    case "check":
      if (args.length !== 1) throw new Error("check accepts no arguments");
      await check();
      console.log("ok");
      break;
    case "docs":
      if (args.length !== 1) throw new Error("docs accepts no arguments");
      await docs();
      console.log("updated documentation");
      break;
    case "fmt":
      if (args.length !== 1) throw new Error("fmt accepts no arguments");
      await format();
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
  console.error(
    `tach: ${typeof value.code === "string" ? `[${value.code}] ` : ""}${
      typeof value.message === "string" ? value.message : String(error)
    }`,
  );
  Deno.exitCode = 1;
}

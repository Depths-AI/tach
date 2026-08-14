#!/usr/bin/env node

import { readFile } from "node:fs/promises";

import { build, check, docs, format, packageVersion } from "./dist/compiler.js";
import { TachError } from "./dist/error.js";

const help = `Tach — lean typed GPGPU compiler

Usage:
  tach build [--target web|spirv]
  tach check
  tach docs
  tach fmt
  tach instructions [--details <section>...]
  tach version
  tach help

Commands:
  build         build the nearest Tach project (web by default)
  check         validate the complete web and SPIR-V project pipeline without writing
  docs          refresh only generated project documentation
  fmt           format every kernel in the nearest Tach project
  instructions  print AI-agent guidance or selected numbered detail sections
  version       print the installed Tach version
  help          print this help
`;

const args = process.argv.slice(2);
try {
  switch (args[0]) {
    case "help": case "-h": case "--help":
      if (args.length !== 1) throw new Error(`${args[0]} accepts no arguments`);
      process.stdout.write(help);
      break;
    case "version":
      if (args.length !== 1) throw new Error("version accepts no arguments");
      console.log(`tach ${await packageVersion()}`);
      break;
    case "instructions": {
      if (args.length !== 1 && (args[1] !== "--details" || args.length < 3)) {
        throw new Error("usage: tach instructions [--details <section>...]");
      }
      const bundle = JSON.parse(await readFile(new URL("./dist/instructions.json", import.meta.url), "utf8"));
      if (args.length === 1) {
        process.stdout.write(`${bundle.mini}\n`);
        break;
      }
      const requested = [...new Set(args.slice(2))];
      const selected = requested.map((number) => {
        if (!/^[1-9]\d*$/u.test(number) || !bundle.sections[number]) {
          throw new Error(`instruction section ${JSON.stringify(number)} does not exist`);
        }
        const section = bundle.sections[number];
        return `## ${number}. ${section.title}\n\n${section.markdown}`;
      });
      process.stdout.write(`${selected.join("\n\n")}\n`);
      break;
    }
    case "build": {
      let target = "web";
      if (args.length === 3 && args[1] === "--target" && (args[2] === "web" || args[2] === "spirv")) target = args[2];
      else if (args.length === 2 && (args[1] === "--target=web" || args[1] === "--target=spirv")) target = args[1].slice("--target=".length);
      else if (args.length !== 1) throw new Error("usage: tach build [--target web|spirv]");
      const result = await build({ target });
      console.log(`built ${result.root} for ${target}`);
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
      throw new Error(args.length === 0 ? "a command is required; run `tach help`" : `unknown command ${JSON.stringify(args[0])}`);
  }
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`tach: ${error instanceof TachError ? `[${error.code}] ` : ""}${message}`);
  process.exitCode = 1;
}

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

function oneLine(value) {
  return String(value ?? "").replaceAll("|", "\\|").replace(/\s+/g, " ").trim();
}

function duration(milliseconds) {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(2)} s`;
}

function annotation(result, type) {
  return result.annotations?.findLast((item) => item.type === type)?.description ?? "";
}

export default class MarkdownReporter {
  constructor(options = {}) {
    this.outputFile = resolve(process.cwd(), options.outputFile ?? "test-report.md");
    this.tests = new Map();
    this.totalTests = 0;
  }

  printsToStdio() {
    return false;
  }

  onBegin(_config, suite) {
    this.totalTests = suite.allTests().length;
  }

  onTestEnd(test, result) {
    this.tests.set(test.id, {
      adapter: annotation(result, "WebGPU adapter"),
      duration: result.duration,
      error: result.error?.message ?? "",
      mode: annotation(result, "WebGPU mode"),
      status: result.status,
      title: test.titlePath().filter(Boolean).join(" › "),
    });
  }

  onEnd(run) {
    const tests = [...this.tests.values()];
    const counts = new Map();
    for (const test of tests) counts.set(test.status, (counts.get(test.status) ?? 0) + 1);
    const webGPU = tests.find((test) => test.mode);
    let cliVersion = "unknown";
    try {
      cliVersion = JSON.parse(readFileSync(resolve(process.cwd(), "build/manifest.json"), "utf8")).cliVersion;
    } catch {
      // The report remains useful even if fixture generation failed before the manifest existed.
    }

    const lines = [
      "# Tach browser test report",
      "",
      `Generated: ${new Date().toISOString()}`,
      "",
      "## Summary",
      "",
      `- Status: **${run.status.toUpperCase()}**`,
      `- Tests: ${tests.length}/${this.totalTests}`,
      `- Passed: ${counts.get("passed") ?? 0}`,
      `- Failed: ${(counts.get("failed") ?? 0) + (counts.get("timedOut") ?? 0)}`,
      `- Skipped: ${counts.get("skipped") ?? 0}`,
      `- Duration: ${duration(run.duration)}`,
      `- Tach CLI: \`${oneLine(cliVersion)}\``,
      `- Host: \`${process.platform}/${process.arch}\`, Node \`${process.version}\``,
      "",
      "## WebGPU execution",
      "",
      webGPU
        ? `- Mode: **${oneLine(webGPU.mode)}**\n- Adapter: \`${oneLine(webGPU.adapter)}\``
        : "- Not exercised by the selected test set.",
      "",
      "## Tests",
      "",
      "| Status | Test | Duration |",
      "| --- | --- | ---: |",
    ];

    const icons = { failed: "❌", interrupted: "⚠️", passed: "✅", skipped: "⏭️", timedOut: "⏱️" };
    for (const test of tests) {
      lines.push(`| ${icons[test.status] ?? test.status} ${test.status} | ${oneLine(test.title)} | ${duration(test.duration)} |`);
    }

    const failures = tests.filter((test) => test.error);
    if (failures.length > 0) {
      lines.push("", "## Failures", "");
      for (const failure of failures) {
        lines.push(
          `### ${oneLine(failure.title)}`,
          "",
          "```text",
          failure.error.replaceAll("```", "'''"),
          "```",
          "",
        );
      }
    }

    mkdirSync(dirname(this.outputFile), { recursive: true });
    writeFileSync(this.outputFile, lines.join("\n") + "\n");
  }
}

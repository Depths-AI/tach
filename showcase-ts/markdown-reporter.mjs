import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

function oneLine(value) {
  return String(value ?? "").replaceAll("|", "\\|").replace(/\s+/gu, " ").trim();
}

function duration(milliseconds) {
  return milliseconds < 1000 ? `${milliseconds.toFixed(1)} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
}

function annotation(result, type) {
  return result.annotations?.findLast((item) => item.type === type)?.description ?? "unknown";
}

function attachedResults(result) {
  const attachment = result.attachments?.find((item) => item.name === "benchmark-results");
  if (!attachment) return undefined;
  const source = attachment.body ?? (attachment.path ? readFileSync(attachment.path) : undefined);
  return source ? JSON.parse(source.toString()) : undefined;
}

export default class MarkdownReporter {
  constructor(options = {}) {
    this.outputFile = options.outputFile;
    this.test = undefined;
  }

  printsToStdio() {
    return false;
  }

  onTestEnd(test, result) {
    this.test = {
      adapter: annotation(result, "WebGPU adapter"),
      profile: annotation(result, "Benchmark profile"),
      duration: result.duration,
      error: result.error?.message ?? "",
      results: attachedResults(result),
      status: result.status,
      title: test.title,
    };
  }

  onEnd(run) {
    const test = this.test;
    const results = test?.results ?? [];
    const verified = results.length === 5 && results.every((item) => item.correct);
    const geometricMean = verified
      ? Math.exp(results.reduce((sum, item) => sum + Math.log(item.speedup), 0) / results.length)
      : undefined;
    const lines = [
      "# Tach TypeScript showcase benchmark report",
      "",
      `Generated: ${new Date().toISOString()}`,
      "",
      "## Summary",
      "",
      `- Status: **${run.status.toUpperCase()}**`,
      `- Profile: **${oneLine(test?.profile)}**`,
      `- Workloads: ${results.length}/5`,
      `- Correctness: **${verified ? "VERIFIED" : "FAILED"}**`,
      `- Geometric-mean acceleration: **${geometricMean === undefined ? "unavailable" : `${geometricMean.toFixed(2)}x`}**`,
      `- Harness duration: ${duration(run.duration)}`,
      `- Adapter: \`${oneLine(test?.adapter)}\``,
      `- Host: \`${process.platform}/${process.arch}\`, Node \`${process.version}\``,
      "",
      "## Measurement contract",
      "",
      "- The native compiler, generated WGSL, shader module, compute pipeline, initial buffer upload, and JavaScript JIT warmup are completed before timing.",
      "- Every sample records multiple dispatches into one compute pass, submits once, and waits once for queue completion.",
      "- Reported times are medians of separate timed batches. GPU readback and correctness comparison happen after timing.",
      "- GPU values are application-visible batch wall times, so command encoding and submission are included; one-time setup and readback are not.",
      "- The comparison target is the same algorithm in single-threaded TypeScript over typed arrays, not a native SIMD or multithreaded library.",
      "",
      "## Results",
      "",
      "| Workload | Problem | Samples | Dispatches/batch | WebGPU median | TypeScript median | Acceleration | WebGPU throughput | Correctness |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ];

    for (const item of results) {
      lines.push(`| ${oneLine(item.name)} | ${oneLine(item.problem)} | ${item.samples} | ${item.dispatches} | ${duration(item.gpuMs)} | ${duration(item.cpuMs)} | **${item.speedup.toFixed(2)}x** | ${item.gpuRate.toFixed(2)} ${oneLine(item.rateUnit)} | ${item.correct ? "PASS" : "FAIL"}: ${oneLine(item.check)} |`);
    }

    if (test?.error) {
      lines.push("", "## Failure", "", "```text", test.error.replaceAll("```", "'''"), "```");
    }
    const outputFile = resolve(
      process.cwd(),
      this.outputFile ?? (test?.profile === "full" ? "benchmark-report.md" : "test-report.md"),
    );
    mkdirSync(dirname(outputFile), { recursive: true });
    writeFileSync(outputFile, lines.join("\n") + "\n");
  }
}

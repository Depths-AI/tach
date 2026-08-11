package spirvtest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"tach/src/compiler"
	"tach/src/spirv"
)

type caseResult struct {
	name       string
	status     string
	duration   time.Duration
	spirvBytes int
	failure    string
}

type markdownReport struct {
	started           time.Time
	duration          time.Duration
	total             int
	results           []caseResult
	setupFailure      string
	adapter           adapter
	adapterKnown      bool
	validationEnabled bool
	validatorVersion  string
}

var report = markdownReport{total: len(exampleCases())}

func TestMain(m *testing.M) {
	report.started = time.Now().UTC()
	started := time.Now()
	code := m.Run()
	report.duration = time.Since(started)
	if err := report.write(filepath.Join(repositoryRoot(), "spirv-test", "test-report.md"), code == 0); err != nil {
		fmt.Fprintln(os.Stderr, "write SPIR-V Markdown report:", err)
		code = 1
	}
	os.Exit(code)
}

func TestExamplesVulkan(t *testing.T) {
	cases := exampleCases()
	if err := checkExampleCoverage(cases); err != nil {
		report.setupFailure = err.Error()
		t.Fatal(err)
	}

	validator, err := exec.LookPath("spirv-val")
	if err != nil {
		report.setupFailure = "spirv-val is not installed"
		t.Fatal(report.setupFailure)
	}
	versionOutput, err := exec.Command(validator, "--version").CombinedOutput()
	if err != nil {
		report.setupFailure = fmt.Sprintf("read spirv-val version: %v", err)
		t.Fatal(report.setupFailure)
	}
	report.validatorVersion = strings.SplitN(strings.TrimSpace(string(versionOutput)), "\n", 2)[0]

	harness, err := openVulkan()
	if err != nil {
		report.setupFailure = err.Error()
		t.Fatal(err)
	}
	report.adapter = harness.adapter
	report.adapterKnown = true
	report.validationEnabled = harness.validationEnabled
	t.Cleanup(harness.close)
	t.Logf("Vulkan execution: %s (%s, %s)", harness.adapter.Mode, harness.adapter.Name, harness.adapter.Type)

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			started := time.Now()
			result := caseResult{name: testCase.name, status: "passed"}
			defer func() {
				result.duration = time.Since(started)
				report.results = append(report.results, result)
			}()

			compilation, err := compiler.CompileFile(filepath.Join(repositoryRoot(), "examples", testCase.name+".tach"))
			if err != nil {
				result.status, result.failure = "failed", err.Error()
				t.Error(err)
				return
			}
			result.spirvBytes = len(compilation.SPIRV)
			if err := validateSPIRV(validator, t.TempDir(), testCase.name, compilation.SPIRV); err != nil {
				result.status, result.failure = "failed", err.Error()
				t.Error(err)
				return
			}
			if _, err := spirv.Summary(compilation.SPIRV); err != nil {
				result.status, result.failure = "failed", err.Error()
				t.Error(err)
				return
			}
			output, err := harness.dispatch(
				compilation.SPIRV,
				compilation.Metadata,
				testCase.kernel,
				testCase.buffers,
				testCase.parameters,
				testCase.invocations,
			)
			if err == nil {
				err = testCase.check(output)
			}
			if err != nil {
				result.status, result.failure = "failed", err.Error()
				t.Error(err)
			}
		})
	}
}

func validateSPIRV(validator, directory, name string, module []byte) error {
	path := filepath.Join(directory, name+".spv")
	if err := os.WriteFile(path, module, 0o600); err != nil {
		return fmt.Errorf("write temporary SPIR-V: %w", err)
	}
	output, err := exec.Command(validator, "--target-env", "vulkan1.1", path).CombinedOutput()
	if err != nil {
		return fmt.Errorf("spirv-val: %w\n%s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

func checkExampleCoverage(cases []exampleCase) error {
	entries, err := filepath.Glob(filepath.Join(repositoryRoot(), "examples", "*.tach"))
	if err != nil {
		return err
	}
	sources := make([]string, len(entries))
	for i, entry := range entries {
		sources[i] = strings.TrimSuffix(filepath.Base(entry), ".tach")
	}
	tests := make([]string, len(cases))
	for i, testCase := range cases {
		tests[i] = testCase.name
	}
	sort.Strings(sources)
	sort.Strings(tests)
	if strings.Join(sources, "\n") != strings.Join(tests, "\n") {
		return fmt.Errorf("SPIR-V cases %v do not exactly cover examples %v", tests, sources)
	}
	return nil
}

func repositoryRoot() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		panic("locate SPIR-V harness source")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), ".."))
}

func (r *markdownReport) write(path string, successful bool) error {
	passed, failed := 0, 0
	for _, result := range r.results {
		if result.status == "passed" {
			passed++
		} else {
			failed++
		}
	}
	if r.setupFailure != "" {
		failed++
	}
	status := "FAILED"
	if successful {
		status = "PASSED"
	}
	validation := "unavailable"
	if r.validationEnabled {
		validation = "enabled (`VK_LAYER_KHRONOS_validation`)"
	}
	lines := []string{
		"# Tach SPIR-V test report",
		"",
		"Generated: " + r.started.Format(time.RFC3339Nano),
		"",
		"## Summary",
		"",
		fmt.Sprintf("- Status: **%s**", status),
		fmt.Sprintf("- Tests: %d/%d", len(r.results), r.total),
		fmt.Sprintf("- Passed: %d", passed),
		fmt.Sprintf("- Failed: %d", failed),
		fmt.Sprintf("- Duration: %s", humanDuration(r.duration)),
		fmt.Sprintf("- Host: `%s/%s`, Go `%s`", runtime.GOOS, runtime.GOARCH, runtime.Version()),
		fmt.Sprintf("- External validation: `%s` (`--target-env vulkan1.1`)", oneLine(r.validatorVersion)),
		"",
		"## Vulkan execution",
		"",
	}
	if r.adapterKnown {
		lines = append(lines,
			fmt.Sprintf("- Mode: **%s**", r.adapter.Mode),
			fmt.Sprintf("- Adapter: `%s`", oneLine(r.adapter.Name)),
			fmt.Sprintf("- Device type: `%s`", r.adapter.Type),
			fmt.Sprintf("- Vulkan API: `%s`", r.adapter.APIVersion),
			fmt.Sprintf("- Vendor/device: `0x%04x/0x%04x`", r.adapter.VendorID, r.adapter.DeviceID),
			"- API validation: "+validation,
		)
	} else {
		lines = append(lines, "- Vulkan was not initialized.")
	}
	lines = append(lines,
		"",
		"## Tests",
		"",
		"| Status | Example | SPIR-V | Duration |",
		"| --- | --- | ---: | ---: |",
	)
	for _, result := range r.results {
		icon := "✅"
		if result.status != "passed" {
			icon = "❌"
		}
		lines = append(lines, fmt.Sprintf(
			"| %s %s | %s | %d bytes | %s |",
			icon, result.status, oneLine(result.name), result.spirvBytes, humanDuration(result.duration),
		))
	}
	if r.setupFailure != "" || failed > 0 {
		lines = append(lines, "", "## Failures", "")
		if r.setupFailure != "" {
			lines = appendFailure(lines, "Harness setup", r.setupFailure)
		}
		for _, result := range r.results {
			if result.failure != "" {
				lines = appendFailure(lines, result.name, result.failure)
			}
		}
	}
	return os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644)
}

func appendFailure(lines []string, title, failure string) []string {
	return append(lines,
		"### "+oneLine(title),
		"",
		"```text",
		strings.ReplaceAll(failure, "```", "'''"),
		"```",
		"",
	)
}

func oneLine(value string) string {
	return strings.TrimSpace(strings.Join(strings.Fields(strings.ReplaceAll(value, "|", "\\|")), " "))
}

func humanDuration(duration time.Duration) string {
	if duration < time.Second {
		return fmt.Sprintf("%d ms", duration.Round(time.Millisecond)/time.Millisecond)
	}
	return fmt.Sprintf("%.2f s", duration.Seconds())
}

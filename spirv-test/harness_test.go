package spirvtest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"tach/src/compiler"
	"tach/src/flow"
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
	compilation, err := compiler.Build(filepath.Join(repositoryRoot(), "examples"), compiler.TargetSPIRV, 0)
	if err != nil {
		report.setupFailure = err.Error()
		t.Fatal(err)
	}
	if err := checkExampleCoverage(cases, compilation); err != nil {
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
	if err := validateSPIRV(validator, t.TempDir(), "kernel", compilation.SPIRV); err != nil {
		report.setupFailure = err.Error()
		t.Fatal(err)
	}
	if _, err := spirv.Summary(compilation.SPIRV); err != nil {
		report.setupFailure = err.Error()
		t.Fatal(err)
	}

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
		t.Run(testCase.name+"/"+testCase.program, func(t *testing.T) {
			started := time.Now()
			result := caseResult{name: testCase.name + "/" + testCase.program, status: "passed"}
			defer func() {
				result.duration = time.Since(started)
				report.results = append(report.results, result)
			}()

			result.spirvBytes = len(compilation.SPIRV)
			output, err := harness.executeProgram(
				compilation.SPIRV,
				compilation.Metadata,
				testCase.program,
				programArguments{Buffers: testCase.buffers, Values: testCase.values, Launch: testCase.launch, Repeat: testCase.repeat},
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

func checkExampleCoverage(cases []exampleCase, compilation *compiler.Result) error {
	covered := map[string]bool{}
	for _, testCase := range cases {
		if covered[testCase.program] {
			return fmt.Errorf("exported program %s has multiple Vulkan cases", testCase.program)
		}
		covered[testCase.program] = true
	}
	for _, program := range compilation.Module.Programs {
		if !covered[program.Name] {
			return fmt.Errorf("exported program %s has no Vulkan case", program.Name)
		}
		delete(covered, program.Name)
	}
	for name := range covered {
		return fmt.Errorf("Vulkan case %s has no exported program", name)
	}
	return nil
}

func TestExampleCoverageRequiresExactlyOneCasePerProgram(t *testing.T) {
	compilation := &compiler.Result{Module: &flow.Module{Programs: []*flow.Program{{Name: "one"}}}}
	for name, cases := range map[string][]exampleCase{
		"missing":   nil,
		"extra":     {{program: "two"}},
		"duplicate": {{program: "one"}, {program: "one"}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := checkExampleCoverage(cases, compilation); err == nil {
				t.Fatal("invalid coverage accepted")
			}
		})
	}
	if err := checkExampleCoverage([]exampleCase{{program: "one"}}, compilation); err != nil {
		t.Fatal(err)
	}
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

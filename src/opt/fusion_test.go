package opt_test

import (
	"strings"
	"testing"

	"tach/src/backend"
	"tach/src/flow"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
)

func fusionModule(t *testing.T, source string) *flow.Module {
	t.Helper()
	a, err := parser.Parse("fusion.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

const identityFusion = `
function produce[i](input: buffer<float32[]>, temporary: buffer<float32[]>) { temporary[i] = input[i] * 2.0; }
function consume[i](temporary: buffer<float32[]>, output: buffer<float32[]>) { output[i] = temporary[i] + 1.0; }
export function transform(input: buffer<float32[]>, output: buffer<float32[]>) {
  const temporary = transient<float32>(min(input.length, output.length));
  run produce(input, temporary) over min(input.length, output.length);
  run consume(temporary, output) over min(input.length, output.length);
}`

func TestIdentityTransientFusionEliminatesDispatchAndStorage(t *testing.T) {
	m := fusionModule(t, identityFusion)
	decision := opt.Decide(m, m.Programs[0], 0, opt.PortablePolicy())
	if !decision.Legal || !decision.Profitable || decision.Reason != "vertical transient forwarding" {
		t.Fatalf("decision = %#v", decision)
	}
	if err := opt.OptimizeLogical(m); err != nil {
		t.Fatal(err)
	}
	if len(m.Programs[0].Dispatches) != 1 || len(m.Programs[0].Resources) != 2 {
		t.Fatalf("optimized Flow IR:\n%s", flow.Dump(m))
	}
}

func TestTargetAffineRecomputeAndCostRejection(t *testing.T) {
	m := fusionModule(t, strings.Replace(identityFusion, "temporary[i] + 1.0", "temporary[i + uint32(1)] + 1.0", 1))
	portable := opt.Decide(m, m.Programs[0], 0, opt.PortablePolicy())
	if portable.Legal {
		t.Fatalf("portable decision = %#v", portable)
	}
	target := opt.Decide(m, m.Programs[0], 0, opt.FusionPolicy{Target: true, MaxInstructions: 4096, MaxLiveValues: 1024, MaxBindings: 8, DispatchBenefit: 100, TransientBenefit: 10, CloneCost: 1})
	if !target.Legal || !target.Profitable || target.Reason != "target affine producer recomputation" {
		t.Fatalf("target decision = %#v", target)
	}
	expensive := opt.Decide(m, m.Programs[0], 0, opt.FusionPolicy{Target: true, MaxInstructions: 4096, MaxLiveValues: 1024, MaxBindings: 8, DispatchBenefit: 1, CloneCost: 100})
	if !expensive.Legal || expensive.Profitable || !strings.Contains(expensive.Reason, "cost") {
		t.Fatalf("expensive decision = %#v", expensive)
	}
	executable, err := backend.Lower(m, backend.WebProfile)
	if err != nil {
		t.Fatal(err)
	}
	if len(executable.Programs[0].Steps) != 1 || len(executable.Programs[0].Transients) != 0 {
		t.Fatalf("target plan = %#v", executable.Programs[0])
	}
}

func TestCrossInvocationExternalDependenceRemainsSeparate(t *testing.T) {
	m := fusionModule(t, `
function increment[i](values: buffer<float32[]>) { values[i] += 1.0; }
function neighbor[i](values: buffer<float32[]>) { values[i] += values[i + uint32(1)]; }
export function graph(values: buffer<float32[]>) { const count = values.length; run increment(values) over count; run neighbor(values) over count; }`)
	decision := opt.Decide(m, m.Programs[0], 0, opt.PortablePolicy())
	if decision.Legal || !strings.Contains(decision.Reason, "cross-invocation") {
		t.Fatalf("decision = %#v", decision)
	}
}

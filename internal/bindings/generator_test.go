package bindings

import (
	"encoding/json"
	"strings"
	"testing"

	"pine/internal/parser"
	"pine/internal/sema"
	"pine/internal/wgsl"
)

func compileBindings(t *testing.T, source string) (*Artifacts, *Metadata) {
	t.Helper()
	a, err := parser.Parse("test.pine", source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	w, err := wgsl.Emit(m)
	if err != nil {
		t.Fatal(err)
	}
	out, err := Generate(m, w)
	if err != nil {
		t.Fatal(err)
	}
	var meta Metadata
	if err := json.Unmarshal(out.MetadataJSON, &meta); err != nil {
		t.Fatal(err)
	}
	return out, &meta
}

func TestScalarUniformUsesPhysicalWrapperSize(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroupSize(32)
export compute scale(data: storage<f32[], read_write>, factor: uniform<f32>) {
  const i = globalId.x;
  if (i < data.length) { data[i] *= factor; }
}`)
	if len(meta.Resources) != 2 {
		t.Fatalf("resources = %d", len(meta.Resources))
	}
	factor := meta.Resources[1]
	if factor.ValueSize != 4 || factor.BindingSize != 16 || factor.MinBindingSize != 16 {
		t.Fatalf("factor sizes = value:%d binding:%d min:%d", factor.ValueSize, factor.BindingSize, factor.MinBindingSize)
	}
	if !strings.Contains(out.JavaScript, "export const byteSize_factor_g0_b1 = 16;") {
		t.Fatal("generated JS does not allocate the physical uniform wrapper size")
	}
}

func TestDirectRuntimeResourceEmitsWriter(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroupSize(1)
export compute clear(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = 0u; }
}`)
	if !strings.Contains(out.JavaScript, "function __write_runtime_u32") {
		t.Fatal("runtime resource writer missing")
	}
}

func TestRuntimeResourcePackerEnforcesMinimumBindingSize(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroupSize(1)
export compute clear(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = 0u; }
}`)
	if meta.Resources[0].MinBindingSize != 4 {
		t.Fatalf("runtime u32 MinBindingSize = %d, want 4", meta.Resources[0].MinBindingSize)
	}
	if !strings.Contains(out.JavaScript, "if (size < 4) throw new RangeError") {
		t.Fatal("runtime resource packer does not enforce Pine's minimum binding size")
	}
}

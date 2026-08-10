package bindings

import (
	"encoding/json"
	"strings"
	"testing"

	"tach/src/parser"
	"tach/src/sema"
	"tach/src/wgsl"
)

func compileBindings(t *testing.T, source string) (*Artifacts, *Metadata) {
	t.Helper()
	a, err := parser.Parse("test.tach", source)
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
	if factor.ByteSize != 16 || factor.MinimumByteSize != 16 {
		t.Fatalf("factor sizes = byte:%d minimum:%d", factor.ByteSize, factor.MinimumByteSize)
	}
	if !strings.Contains(out.JavaScript, `"name":"factor"`) || !strings.Contains(out.JavaScript, `"byteSize":16`) {
		t.Fatal("generated JS does not carry the physical uniform wrapper size into its private resource descriptor")
	}
}

func TestDirectRuntimeResourceCarriesHostLayout(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroupSize(1)
export compute clear(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = 0u; }
}`)
	if !strings.Contains(out.JavaScript, `"kind":"runtime"`) ||
		!strings.Contains(out.JavaScript, `from "@depths/tach/internal"`) {
		t.Fatal("runtime compute-buffer descriptor missing")
	}
}

func TestAtomicResourceUsesUnderlyingHostRepresentation(t *testing.T) {
	out, _ := compileBindings(t, `
type Counters = { total: atomic<u32> };
@workgroupSize(1)
export compute increment(counters: storage<Counters, read_write>) {
  atomicAdd(counters.total, 1u);
}`)
	if !strings.Contains(out.JavaScript, `"name":"total","offset":0,"type":{"kind":"u32"`) {
		t.Fatal("atomic resource does not use its underlying u32 host representation")
	}
}

func TestRuntimeResourceDescriptorRecordsMinimumBindingSize(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroupSize(1)
export compute clear(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = 0u; }
}`)
	if meta.Resources[0].MinimumByteSize != 4 {
		t.Fatalf("runtime u32 MinimumByteSize = %d, want 4", meta.Resources[0].MinimumByteSize)
	}
	if !strings.Contains(out.JavaScript, `"minimumByteSize":4`) {
		t.Fatal("runtime resource descriptor omits Tach's minimum binding size")
	}
}

func TestGeneratedSurfaceMirrorsSource(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroupSize(64)
export compute scale(data: storage<f32[], read_write>, factor: uniform<f32>) {
  const i = globalId.x;
  if (i < data.length) { data[i] *= factor; }
}`)
	if len(meta.Kernels) != 1 || meta.Kernels[0].Name != "scale" || meta.Kernels[0].EntryPoint != "scale" {
		t.Fatalf("kernel metadata = %+v", meta.Kernels)
	}
	for _, want := range []string{
		"import { defineModule as $defineModule } from \"@depths/tach/internal\"",
		"const $tach = $defineModule({",
		"export function scale(data, factor, $dispatch)",
		"return $tach.dispatch(0",
		"import type { ComputeBuffer, ComputeDispatch, DispatchOptions } from \"@depths/tach\"",
		"data: ComputeBuffer<Float32Array | readonly number[]>",
		"factor: number",
		"$dispatch?: DispatchOptions",
		"): ComputeDispatch",
	} {
		if !strings.Contains(out.JavaScript+out.Declarations, want) {
			t.Fatalf("generated bindings missing %q", want)
		}
	}
}

func TestKernelMayBeNamedBuffer(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroupSize(1)
export compute buffer(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = 0u; }
}`)
	if !strings.Contains(out.JavaScript, "export function buffer(data, $dispatch)") ||
		!strings.Contains(out.Declarations, "export function buffer(") {
		t.Fatal("source kernel named buffer was not preserved")
	}
}

func TestScalarRuntimeArraysExposeTypedHostRepresentations(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroupSize(1)
export compute arrays(
  signed: storage<i32[], read_write>,
  unsigned: storage<u32[], read_write>,
  floats: storage<f32[], read_write>,
) { }
`)
	for _, want := range []string{
		"signed: ComputeBuffer<Int32Array | readonly number[]>",
		"unsigned: ComputeBuffer<Uint32Array | readonly number[]>",
		"floats: ComputeBuffer<Float32Array | readonly number[]>",
	} {
		if !strings.Contains(out.Declarations, want) {
			t.Fatalf("generated declarations missing %q", want)
		}
	}
}

func TestSourceOwnedTachPrefixIsNotReserved(t *testing.T) {
	out, _ := compileBindings(t, `
type TachBuffer = { value: u32 };
@workgroupSize(1)
export compute preserve(data: storage<TachBuffer, read_write>) { }
`)
	if !strings.Contains(out.Declarations, "export interface TachBuffer") ||
		!strings.Contains(out.Declarations, "data: ComputeBuffer<TachBuffer>") {
		t.Fatal("source-owned TachBuffer name was not preserved exactly")
	}
}

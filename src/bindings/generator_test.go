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
@workgroup(32)
export compute scale[i](data: buffer<f32[]>, factor: uniform<f32>) {
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
@workgroup(1)
export compute clear[i](data: buffer<u32[]>) {
  if (i < data.length) { data[i] = 0; }
}`)
	if !strings.Contains(out.JavaScript, `"kind":"runtime"`) ||
		!strings.Contains(out.JavaScript, `from "@depths/tach/internal"`) {
		t.Fatal("runtime compute-buffer descriptor missing")
	}
}

func TestAtomicResourceUsesUnderlyingHostRepresentation(t *testing.T) {
	out, _ := compileBindings(t, `
type Counters = { total: atomic<u32> };
@workgroup(1)
export compute increment[i](counters: buffer<Counters>) {
  atomicAdd(counters.total, 1);
}`)
	if !strings.Contains(out.JavaScript, `"name":"total","offset":0,"type":{"kind":"u32"`) {
		t.Fatal("atomic resource does not use its underlying u32 host representation")
	}
}

func TestRuntimeResourceDescriptorRecordsMinimumBindingSize(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroup(1)
export compute clear[i](data: buffer<u32[]>) {
  if (i < data.length) { data[i] = 0; }
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
@workgroup(64)
export compute scale[i](data: buffer<f32[]>, factor: uniform<f32>) {
  if (i < data.length) { data[i] *= factor; }
}`)
	if len(meta.Kernels) != 1 || meta.Kernels[0].Name != "scale" || meta.Kernels[0].EntryPoint != "scale" || meta.Kernels[0].Dimensions != 1 {
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
		"$dispatch?: DispatchOptions<number>",
		"): ComputeDispatch",
	} {
		if !strings.Contains(out.JavaScript+out.Declarations, want) {
			t.Fatalf("generated bindings missing %q", want)
		}
	}
}

func TestKernelMayBeNamedBuffer(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroup(1)
export compute buffer[i](data: buffer<u32[]>) {
  if (i < data.length) { data[i] = 0; }
}`)
	if !strings.Contains(out.JavaScript, "export function buffer(data, $dispatch)") ||
		!strings.Contains(out.Declarations, "export function buffer(") {
		t.Fatal("source kernel named buffer was not preserved")
	}
}

func TestPackedRuntimeArraysExposeTypedHostRepresentations(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroup(1)
export compute arrays[i](
  signed: buffer<i32[]>,
  unsigned: buffer<u32[]>,
  floats: buffer<f32[]>,
  vectors: buffer<f32x4[]>,
) { }
`)
	for _, want := range []string{
		"signed: ComputeBuffer<Int32Array | readonly number[]>",
		"unsigned: ComputeBuffer<Uint32Array | readonly number[]>",
		"floats: ComputeBuffer<Float32Array | readonly number[]>",
		"vectors: ComputeBuffer<Float32Array | ReadonlyArray<readonly [number, number, number, number]>>",
	} {
		if !strings.Contains(out.Declarations, want) {
			t.Fatalf("generated declarations missing %q", want)
		}
	}
}

func TestSourceOwnedTachPrefixIsNotReserved(t *testing.T) {
	out, _ := compileBindings(t, `
type TachBuffer = { value: u32 };
@workgroup(1)
export compute preserve[i](data: buffer<TachBuffer>) { }
`)
	if !strings.Contains(out.Declarations, "export interface TachBuffer") ||
		!strings.Contains(out.Declarations, "data: ComputeBuffer<TachBuffer>") {
		t.Fatal("source-owned TachBuffer name was not preserved exactly")
	}
}

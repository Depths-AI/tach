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

func TestValuesShareOnePhysicalParameterBlock(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroup(32)
export function scale[i](data: buffer<float32[]>, factor: float32, enabled: bool) {
  if (enabled && i < data.length) { data[i] *= factor; }
}`)
	if len(meta.Resources) != 1 {
		t.Fatalf("resources = %d", len(meta.Resources))
	}
	block := meta.Kernels[0].ParameterBlock
	if block == nil || block.Binding != 1 || block.ByteSize != 16 || len(block.Fields) != 2 {
		t.Fatalf("parameter block = %+v", block)
	}
	if !strings.Contains(out.JavaScript, `"parameterBlock":{"group":0,"binding":1,"byteSize":16`) ||
		!strings.Contains(out.JavaScript, `"kind":"bool"`) {
		t.Fatal("generated JS does not carry the packed value block and bool codec")
	}
}

func TestDirectRuntimeResourceCarriesHostLayout(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroup(1)
export function clear[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = 0; }
}`)
	if !strings.Contains(out.JavaScript, `"kind":"runtime"`) ||
		!strings.Contains(out.JavaScript, `from "@depths/tach/internal"`) {
		t.Fatal("runtime compute-buffer descriptor missing")
	}
}

func TestAtomicResourceUsesUnderlyingHostRepresentation(t *testing.T) {
	out, _ := compileBindings(t, `
type Counters = { total: atomic<uint32> };
@workgroup(1)
export function increment[i](counters: buffer<Counters>) {
  atomicAdd(counters.total, 1);
}`)
	if !strings.Contains(out.JavaScript, `"name":"total","offset":0,"type":{"kind":"u32"`) {
		t.Fatal("atomic resource does not use its underlying uint32 host representation")
	}
}

func TestRuntimeResourceDescriptorRecordsMinimumBindingSize(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroup(1)
export function clear[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = 0; }
}`)
	if meta.Resources[0].MinimumByteSize != 4 {
		t.Fatalf("runtime uint32 MinimumByteSize = %d, want 4", meta.Resources[0].MinimumByteSize)
	}
	if !strings.Contains(out.JavaScript, `"minimumByteSize":4`) {
		t.Fatal("runtime resource descriptor omits Tach's minimum binding size")
	}
}

func TestGeneratedSurfaceMirrorsSource(t *testing.T) {
	out, meta := compileBindings(t, `
@workgroup(64)
export function scale[i](data: buffer<float32[]>, factor: float32) {
  if (i < data.length) { data[i] *= factor; }
}`)
	if len(meta.Kernels) != 1 || meta.Kernels[0].Name != "scale" || meta.Kernels[0].EntryPoint != "scale" || meta.Kernels[0].Dimensions != 1 {
		t.Fatalf("kernel metadata = %+v", meta.Kernels)
	}
	for _, want := range []string{
		"import { defineModule as $defineModule } from \"@depths/tach/internal\"",
		"const $tach = $defineModule({",
		"export function scale(data, factor, $launch)",
		"return $tach.command(0",
		"import type { ComputeBuffer, ComputeCommand, LaunchOptions } from \"@depths/tach\"",
		"data: ComputeBuffer<Float32Array | readonly number[]>",
		"factor: number",
		"$launch?: LaunchOptions<number>",
		"): ComputeCommand",
	} {
		if !strings.Contains(out.JavaScript+out.Declarations, want) {
			t.Fatalf("generated bindings missing %q", want)
		}
	}
}

func TestKernelMayBeNamedBuffer(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroup(1)
export function buffer[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = 0; }
}`)
	if !strings.Contains(out.JavaScript, "export function buffer(data, $launch)") ||
		!strings.Contains(out.Declarations, "export function buffer(") {
		t.Fatal("source kernel named buffer was not preserved")
	}
}

func TestPackedRuntimeArraysExposeTypedHostRepresentations(t *testing.T) {
	out, _ := compileBindings(t, `
@workgroup(1)
export function arrays[i](
  signed: buffer<int32[]>,
  unsigned: buffer<uint32[]>,
  floats: buffer<float32[]>,
  vectors: buffer<float32x4[]>,
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
type TachBuffer = { value: uint32 };
@workgroup(1)
export function preserve[i](data: buffer<TachBuffer>) { }
`)
	if !strings.Contains(out.Declarations, "export type TachBuffer") ||
		!strings.Contains(out.Declarations, "data: ComputeBuffer<TachBuffer>") {
		t.Fatal("source-owned TachBuffer name was not preserved exactly")
	}
}

package foundation

import (
	"math"
	"testing"
)

func TestDiagnosticsSortWithoutMutatingInput(t *testing.T) {
	diagnostics := Diagnostics{
		{Kind: "z", Severity: "warning", Span: Span{File: "b.tach", Start: Position{Offset: 2, Line: 1, Column: 3}}, Message: "last"},
		{Kind: "b", Severity: "warning", Span: Span{File: "a.tach", Start: Position{Offset: 1, Line: 1, Column: 2}}, Message: "warning"},
		{Kind: "a", Span: Span{File: "a.tach", Start: Position{Offset: 1, Line: 1, Column: 2}}, Message: "error"},
	}

	sorted := diagnostics.Sorted()
	if sorted[0].Message != "error" || sorted[1].Message != "warning" || sorted[2].Message != "last" {
		t.Fatalf("diagnostic order = %q, %q, %q", sorted[0].Message, sorted[1].Message, sorted[2].Message)
	}
	if diagnostics[0].Message != "last" {
		t.Fatal("sorting mutated the input diagnostics")
	}
	if got := sorted.Error(); got != "a.tach:1:2: error\na.tach:1:2: warning\nb.tach:1:3: last" {
		t.Fatalf("formatted diagnostics = %q", got)
	}
}

func TestTypeDomainsRemainDistinct(t *testing.T) {
	plain := &Type{Kind: StructKind, Name: "Plain", Fields: []TypeField{{Name: "value", Type: VectorOf(Float32Type, 2)}}}
	mask := VectorOf(BoolType, 2)
	atomic := AtomicOf(Uint32Type)
	runtime := RuntimeArrayOf(Float32Type)

	if !IsConstructible(plain) || !IsHostShareable(plain) || !IsHostParameter(plain) || !IsWorkgroupStorable(plain) || !IsTransientElement(plain) {
		t.Fatal("plain numeric structure must be valid in every value and storage domain")
	}
	if !IsConstructible(mask) || IsHostShareable(mask) || IsHostParameter(mask) || IsWorkgroupStorable(mask) || IsTransientElement(mask) {
		t.Fatal("boolean vectors must remain value-only")
	}
	if IsConstructible(atomic) || !IsHostShareable(atomic) || !IsWorkgroupStorable(atomic) || IsTransientElement(atomic) {
		t.Fatal("atomic values must remain storage-only and non-transient")
	}
	if IsConstructible(runtime) || !IsHostShareable(runtime) || IsHostParameter(runtime) || IsWorkgroupStorable(runtime) || IsTransientElement(runtime) {
		t.Fatal("runtime arrays must remain buffer-only")
	}
}

func TestFloat16EncodingBoundaries(t *testing.T) {
	for _, test := range []struct {
		value float64
		bits  uint16
	}{
		{0, 0x0000},
		{-2, 0xc000},
		{0x1p-24, 0x0001},
		{1 + 0x1p-11, 0x3c00},
		{65504, 0x7bff},
	} {
		bits, ok := Float16Bits(test.value)
		if !ok || bits != test.bits {
			t.Errorf("Float16Bits(%g) = %#04x, %v; want %#04x, true", test.value, bits, ok, test.bits)
		}
	}
	for _, value := range []float64{65505, math.Inf(1), math.NaN()} {
		if _, ok := Float16Bits(value); ok {
			t.Errorf("Float16Bits(%g) accepted an unrepresentable value", value)
		}
	}
	if (&ConstantValue{Type: Float16Type, Bits: []uint32{0x7c00}}).Valid() {
		t.Fatal("infinite float16 constant must be invalid")
	}
}

func TestPortableStructLayout(t *testing.T) {
	particle := &Type{
		Kind: StructKind,
		Name: "Particle",
		Fields: []TypeField{
			{Name: "position", Type: VectorOf(Float32Type, 3)},
			{Name: "mass", Type: Float32Type},
			{Name: "velocity", Type: VectorOf(Float32Type, 3)},
		},
	}

	l, err := LayoutOf(particle)
	if err != nil {
		t.Fatal(err)
	}
	if l.Size != 32 || l.Align != 16 || l.Runtime {
		t.Fatalf("layout = size %d align %d runtime %v, want size 32 align 16 runtime false", l.Size, l.Align, l.Runtime)
	}
	wantOffsets := []uint32{0, 12, 16}
	for i, want := range wantOffsets {
		if got := l.Fields[i].Offset; got != want {
			t.Fatalf("field %s offset = %d, want %d", l.Fields[i].Name, got, want)
		}
	}
}

func TestNestedStructReservesSixteenByteBoundary(t *testing.T) {
	inner := &Type{
		Kind:   StructKind,
		Name:   "Inner",
		Fields: []TypeField{{Name: "x", Type: Float32Type}},
	}
	outer := &Type{
		Kind: StructKind,
		Name: "Outer",
		Fields: []TypeField{
			{Name: "before", Type: Float32Type},
			{Name: "inner", Type: inner},
			{Name: "after", Type: Float32Type},
		},
	}

	l, err := LayoutOf(outer)
	if err != nil {
		t.Fatal(err)
	}
	if l.Size != 48 || l.Align != 16 {
		t.Fatalf("outer layout = size %d align %d, want size 48 align 16", l.Size, l.Align)
	}
	wantOffsets := []uint32{0, 16, 32}
	for i, want := range wantOffsets {
		if got := l.Fields[i].Offset; got != want {
			t.Fatalf("field %s offset = %d, want %d", l.Fields[i].Name, got, want)
		}
	}
}

func TestRuntimeTailLayout(t *testing.T) {
	block := &Type{
		Kind: StructKind,
		Name: "Block",
		Fields: []TypeField{
			{Name: "count", Type: Uint32Type},
			{Name: "values", Type: RuntimeArrayOf(VectorOf(Float32Type, 3))},
		},
	}

	l, err := LayoutOf(block)
	if err != nil {
		t.Fatal(err)
	}
	if !l.Runtime || l.Size != 16 || l.Align != 16 {
		t.Fatalf("runtime block = size %d align %d runtime %v, want size 16 align 16 runtime true", l.Size, l.Align, l.Runtime)
	}
	values := l.Fields[1]
	if values.Offset != 16 || !values.Layout.Runtime || values.Layout.Stride != 16 {
		t.Fatalf("runtime tail = offset %d stride %d runtime %v, want offset 16 stride 16 runtime true", values.Offset, values.Layout.Stride, values.Layout.Runtime)
	}
}

func TestFixedArrayStride(t *testing.T) {
	l, err := LayoutOf(FixedArrayOf(VectorOf(Float32Type, 3), 3))
	if err != nil {
		t.Fatal(err)
	}
	if l.Size != 48 || l.Align != 16 || l.Stride != 16 {
		t.Fatalf("array layout = size %d align %d stride %d, want 48/16/16", l.Size, l.Align, l.Stride)
	}
}

func TestFloat16Layout(t *testing.T) {
	for _, test := range []struct {
		typ                 *Type
		size, align, stride uint32
	}{
		{Float16Type, 2, 2, 0},
		{VectorOf(Float16Type, 2), 4, 4, 0},
		{VectorOf(Float16Type, 3), 6, 8, 0},
		{VectorOf(Float16Type, 4), 8, 8, 0},
		{RuntimeArrayOf(VectorOf(Float16Type, 3)), 0, 8, 8},
	} {
		got, err := LayoutOf(test.typ)
		if err != nil {
			t.Fatal(err)
		}
		if got.Size != test.size || got.Align != test.align || got.Stride != test.stride {
			t.Errorf("%s layout = %d/%d/%d, want %d/%d/%d", test.typ, got.Size, got.Align, got.Stride, test.size, test.align, test.stride)
		}
	}
}

func TestFixedArraySizeCannotOverflow(t *testing.T) {
	if _, err := LayoutOf(FixedArrayOf(Uint32Type, ^uint32(0))); err == nil {
		t.Fatal("expected an overflowing fixed array layout to be rejected")
	}
}

func TestBoolHasNoHostLayout(t *testing.T) {
	if _, err := LayoutOf(BoolType); err == nil {
		t.Fatal("expected bool host layout to be rejected")
	}
	if _, err := LayoutOf(VectorOf(BoolType, 4)); err == nil {
		t.Fatal("expected boolean vector host layout to be rejected")
	}
}

func TestRuntimeArrayMustBeFinal(t *testing.T) {
	bad := &Type{
		Kind: StructKind,
		Name: "Bad",
		Fields: []TypeField{
			{Name: "values", Type: RuntimeArrayOf(Uint32Type)},
			{Name: "trailer", Type: Uint32Type},
		},
	}
	if _, err := LayoutOf(bad); err == nil {
		t.Fatal("expected non-final runtime array to be rejected")
	}
}

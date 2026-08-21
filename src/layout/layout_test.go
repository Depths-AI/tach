package layout

import (
	"testing"

	"tach/src/types"
)

func TestPortableStructLayout(t *testing.T) {
	particle := &types.Type{
		Kind: types.Struct,
		Name: "Particle",
		Fields: []types.Field{
			{Name: "position", Type: types.Vec(types.TF32, 3)},
			{Name: "mass", Type: types.TF32},
			{Name: "velocity", Type: types.Vec(types.TF32, 3)},
		},
	}

	l, err := Of(particle)
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
	inner := &types.Type{
		Kind:   types.Struct,
		Name:   "Inner",
		Fields: []types.Field{{Name: "x", Type: types.TF32}},
	}
	outer := &types.Type{
		Kind: types.Struct,
		Name: "Outer",
		Fields: []types.Field{
			{Name: "before", Type: types.TF32},
			{Name: "inner", Type: inner},
			{Name: "after", Type: types.TF32},
		},
	}

	l, err := Of(outer)
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
	block := &types.Type{
		Kind: types.Struct,
		Name: "Block",
		Fields: []types.Field{
			{Name: "count", Type: types.TU32},
			{Name: "values", Type: types.Runtime(types.Vec(types.TF32, 3))},
		},
	}

	l, err := Of(block)
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
	l, err := Of(types.Array(types.Vec(types.TF32, 3), 3))
	if err != nil {
		t.Fatal(err)
	}
	if l.Size != 48 || l.Align != 16 || l.Stride != 16 {
		t.Fatalf("array layout = size %d align %d stride %d, want 48/16/16", l.Size, l.Align, l.Stride)
	}
}

func TestFloat16Layout(t *testing.T) {
	for _, test := range []struct {
		type_               *types.Type
		size, align, stride uint32
	}{
		{types.TF16, 2, 2, 0},
		{types.Vec(types.TF16, 2), 4, 4, 0},
		{types.Vec(types.TF16, 3), 6, 8, 0},
		{types.Vec(types.TF16, 4), 8, 8, 0},
		{types.Runtime(types.Vec(types.TF16, 3)), 0, 8, 8},
	} {
		got, err := Of(test.type_)
		if err != nil {
			t.Fatal(err)
		}
		if got.Size != test.size || got.Align != test.align || got.Stride != test.stride {
			t.Errorf("%s layout = %d/%d/%d, want %d/%d/%d", test.type_, got.Size, got.Align, got.Stride, test.size, test.align, test.stride)
		}
	}
}

func TestFixedArraySizeCannotOverflow(t *testing.T) {
	if _, err := Of(types.Array(types.TU32, ^uint32(0))); err == nil {
		t.Fatal("expected an overflowing fixed array layout to be rejected")
	}
}

func TestBoolHasNoHostLayout(t *testing.T) {
	if _, err := Of(types.TBool); err == nil {
		t.Fatal("expected bool host layout to be rejected")
	}
	if _, err := Of(types.Vec(types.TBool, 4)); err == nil {
		t.Fatal("expected boolean vector host layout to be rejected")
	}
}

func TestRuntimeArrayMustBeFinal(t *testing.T) {
	bad := &types.Type{
		Kind: types.Struct,
		Name: "Bad",
		Fields: []types.Field{
			{Name: "values", Type: types.Runtime(types.TU32)},
			{Name: "trailer", Type: types.TU32},
		},
	}
	if _, err := Of(bad); err == nil {
		t.Fatal("expected non-final runtime array to be rejected")
	}
}

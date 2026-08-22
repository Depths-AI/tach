// Package foundation defines the source, type, constant, and host-layout
// primitives shared by every higher compiler layer.
package foundation

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

type Position struct {
	Offset int `json:"offset"`
	Line   int `json:"line"`
	Column int `json:"column"`
}

type Span struct {
	File  string   `json:"file"`
	Start Position `json:"start"`
	End   Position `json:"end"`
}

func (s Span) String() string {
	if s.File == "" {
		return fmt.Sprintf("%d:%d", s.Start.Line, s.Start.Column)
	}
	return fmt.Sprintf("%s:%d:%d", s.File, s.Start.Line, s.Start.Column)
}

type RelatedDiagnostic struct {
	Span    Span   `json:"span"`
	Message string `json:"message"`
	Source  string `json:"source,omitempty"`
}

type Diagnostic struct {
	Severity string              `json:"severity"`
	Kind     string              `json:"code"`
	Span     Span                `json:"span"`
	Message  string              `json:"message"`
	Help     string              `json:"help,omitempty"`
	Source   string              `json:"source,omitempty"`
	Related  []RelatedDiagnostic `json:"related,omitempty"`
}

func (d Diagnostic) Error() string { return fmt.Sprintf("%s: %s", d.Span.String(), d.Message) }

type Diagnostics []Diagnostic

func (ds Diagnostics) Error() string {
	parts := make([]string, len(ds))
	for i := range ds {
		parts[i] = ds[i].Error()
	}
	return strings.Join(parts, "\n")
}

func (ds Diagnostics) Sorted() Diagnostics {
	out := append(Diagnostics(nil), ds...)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Span.File != b.Span.File {
			return a.Span.File < b.Span.File
		}
		if a.Span.Start.Offset != b.Span.Start.Offset {
			return a.Span.Start.Offset < b.Span.Start.Offset
		}
		if a.Severity != b.Severity {
			return a.Severity != "warning"
		}
		return a.Kind < b.Kind
	})
	return out
}

type TypeKind uint8

const (
	InvalidKind TypeKind = iota
	VoidKind
	BoolKind
	Int32Kind
	Uint32Kind
	Float16Kind
	Float32Kind
	VectorKind
	StructKind
	AtomicKind
	FixedArrayKind
	RuntimeArrayKind
)

type Type struct {
	Kind   TypeKind
	Elem   *Type
	Lanes  int
	Name   string
	Fields []TypeField
	Count  uint32
}

type TypeField struct {
	Name string
	Type *Type
}

// ConstantValue is a fully evaluated compile-time scalar or vector. Bits
// contains one canonical lane for scalars and one entry per vector lane.
type ConstantValue struct {
	Type *Type
	Bits []uint32
}

func IsConstantType(t *Type) bool {
	return IsScalar(t) || t != nil && t.Kind == VectorKind && IsScalar(t.Elem)
}

func (v *ConstantValue) Valid() bool {
	if v == nil || !IsConstantType(v.Type) {
		return false
	}
	lanes, element := 1, v.Type
	if v.Type.Kind == VectorKind {
		lanes, element = v.Type.Lanes, v.Type.Elem
	}
	if len(v.Bits) != lanes {
		return false
	}
	for _, bits := range v.Bits {
		if element.Kind == BoolKind && bits > 1 || element.Kind == Float16Kind && (bits > 0xffff || !finite(Float16FromBits(uint16(bits)))) || element.Kind == Float32Kind && !finite(float64(math.Float32frombits(bits))) {
			return false
		}
	}
	return true
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

var (
	VoidType    = &Type{Kind: VoidKind, Name: "void"}
	BoolType    = &Type{Kind: BoolKind, Name: "bool"}
	Int32Type   = &Type{Kind: Int32Kind, Name: "int32"}
	Uint32Type  = &Type{Kind: Uint32Kind, Name: "uint32"}
	Float16Type = &Type{Kind: Float16Kind, Name: "float16"}
	Float32Type = &Type{Kind: Float32Kind, Name: "float32"}
)

func VectorOf(elem *Type, lanes int) *Type {
	return &Type{Kind: VectorKind, Elem: elem, Lanes: lanes}
}
func AtomicOf(elem *Type) *Type { return &Type{Kind: AtomicKind, Elem: elem} }
func FixedArrayOf(elem *Type, count uint32) *Type {
	return &Type{Kind: FixedArrayKind, Elem: elem, Count: count}
}
func RuntimeArrayOf(elem *Type) *Type { return &Type{Kind: RuntimeArrayKind, Elem: elem} }

func (t *Type) String() string {
	if t == nil {
		return "<nil>"
	}
	switch t.Kind {
	case VoidKind, BoolKind, Int32Kind, Uint32Kind, Float16Kind, Float32Kind:
		return t.Name
	case VectorKind:
		return fmt.Sprintf("vec<%s, %d>", t.Elem, t.Lanes)
	case StructKind:
		return t.Name
	case AtomicKind:
		return fmt.Sprintf("atomic<%s>", t.Elem)
	case FixedArrayKind:
		return fmt.Sprintf("%s[%d]", t.Elem, t.Count)
	case RuntimeArrayKind:
		return fmt.Sprintf("%s[]", t.Elem)
	default:
		return "<invalid>"
	}
}

func Equal(a, b *Type) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil || a.Kind != b.Kind {
		return false
	}
	switch a.Kind {
	case VoidKind, BoolKind, Int32Kind, Uint32Kind, Float16Kind, Float32Kind:
		return true
	case VectorKind:
		return a.Lanes == b.Lanes && Equal(a.Elem, b.Elem)
	case StructKind:
		return a.Name == b.Name
	case AtomicKind:
		return Equal(a.Elem, b.Elem)
	case FixedArrayKind:
		return a.Count == b.Count && Equal(a.Elem, b.Elem)
	case RuntimeArrayKind:
		return Equal(a.Elem, b.Elem)
	}
	return false
}

func IsScalar(t *Type) bool {
	return t != nil && (t.Kind == Int32Kind || t.Kind == Uint32Kind || t.Kind == Float16Kind || t.Kind == Float32Kind || t.Kind == BoolKind)
}
func IsNumericScalar(t *Type) bool {
	return t != nil && (t.Kind == Int32Kind || t.Kind == Uint32Kind || t.Kind == Float16Kind || t.Kind == Float32Kind)
}
func IsNumeric(t *Type) bool {
	return IsNumericScalar(t) || (t != nil && t.Kind == VectorKind && IsNumericScalar(t.Elem))
}
func IsBoolean(t *Type) bool {
	return t != nil && (t.Kind == BoolKind || t.Kind == VectorKind && t.Elem != nil && t.Elem.Kind == BoolKind)
}
func BoolShape(t *Type) *Type {
	if t != nil && t.Kind == VectorKind {
		return VectorOf(BoolType, t.Lanes)
	}
	return BoolType
}
func IsInteger(t *Type) bool { return t != nil && (t.Kind == Int32Kind || t.Kind == Uint32Kind) }
func IsIntegerLike(t *Type) bool {
	return IsInteger(t) || t != nil && t.Kind == VectorKind && IsInteger(t.Elem)
}
func IsFloatLike(t *Type) bool {
	return t != nil && (t.Kind == Float16Kind || t.Kind == Float32Kind || t.Kind == VectorKind && t.Elem != nil && (t.Elem.Kind == Float16Kind || t.Elem.Kind == Float32Kind))
}
func ShiftCountType(t *Type) *Type {
	if IsInteger(t) {
		return Uint32Type
	}
	if t != nil && t.Kind == VectorKind && IsInteger(t.Elem) {
		return VectorOf(Uint32Type, t.Lanes)
	}
	return nil
}

// IsSignedNumeric reports whether unary negation is well-defined in Tach's
// portable numeric profile. Unsigned integers intentionally have no unary -.
func IsSignedNumeric(t *Type) bool {
	if t == nil {
		return false
	}
	if t.Kind == Int32Kind || t.Kind == Float16Kind || t.Kind == Float32Kind {
		return true
	}
	return t.Kind == VectorKind && IsSignedNumeric(t.Elem)
}

func Contains(t *Type, kind TypeKind) bool {
	if t == nil {
		return false
	}
	if t.Kind == kind {
		return true
	}
	switch t.Kind {
	case VectorKind, AtomicKind, FixedArrayKind, RuntimeArrayKind:
		return Contains(t.Elem, kind)
	case StructKind:
		for _, f := range t.Fields {
			if Contains(f.Type, kind) {
				return true
			}
		}
	}
	return false
}

// ContainsRuntimeArray is true when the type has a runtime-sized tail, directly
// or through a nested structure. Such values can be addressed in buffers but
// cannot be loaded, constructed, passed by value, or used in parameter blocks.
func ContainsRuntimeArray(t *Type) bool { return Contains(t, RuntimeArrayKind) }

// IsConstructible is the value-domain counterpart to IsHostShareable.
func IsConstructible(t *Type) bool {
	if t == nil || ContainsRuntimeArray(t) {
		return false
	}
	switch t.Kind {
	case BoolKind, Int32Kind, Uint32Kind, Float16Kind, Float32Kind:
		return true
	case VectorKind:
		return IsConstructible(t.Elem)
	case StructKind:
		for _, f := range t.Fields {
			if !IsConstructible(f.Type) {
				return false
			}
		}
		return true
	}
	return false
}
func IsHostShareable(t *Type) bool {
	if t == nil {
		return false
	}
	switch t.Kind {
	case Int32Kind, Uint32Kind, Float16Kind, Float32Kind:
		return true
	case VectorKind:
		return IsHostShareable(t.Elem)
	case AtomicKind:
		return t.Elem.Kind == Int32Kind || t.Elem.Kind == Uint32Kind
	case FixedArrayKind:
		return false // fixed arrays are currently a workgroup-memory type in Tach
	case StructKind:
		for _, f := range t.Fields {
			if !IsHostShareable(f.Type) {
				return false
			}
		}
		return true
	case RuntimeArrayKind:
		return IsHostShareable(t.Elem)
	}
	return false
}
func IsHostParameter(t *Type) bool {
	if t == nil {
		return false
	}
	if t.Kind == BoolKind || IsNumeric(t) {
		return true
	}
	if t.Kind == StructKind {
		for _, field := range t.Fields {
			if !IsHostParameter(field.Type) {
				return false
			}
		}
		return true
	}
	return false
}
func FieldIndex(t *Type, name string) int {
	if t == nil || t.Kind != StructKind {
		return -1
	}
	for i, f := range t.Fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

func ParseBuiltin(name string) *Type {
	switch name {
	case "void":
		return VoidType
	case "bool":
		return BoolType
	case "int32":
		return Int32Type
	case "uint32":
		return Uint32Type
	case "float16":
		return Float16Type
	case "float32":
		return Float32Type
	}
	return nil
}

func ContainsAtomic(t *Type) bool { return Contains(t, AtomicKind) }

func IsWorkgroupStorable(t *Type) bool {
	if t == nil || ContainsRuntimeArray(t) {
		return false
	}
	switch t.Kind {
	case Int32Kind, Uint32Kind, Float16Kind, Float32Kind:
		return true
	case VectorKind:
		return IsWorkgroupStorable(t.Elem)
	case AtomicKind:
		return t.Elem.Kind == Int32Kind || t.Elem.Kind == Uint32Kind
	case FixedArrayKind:
		return t.Count > 0 && IsWorkgroupStorable(t.Elem)
	case StructKind:
		for _, f := range t.Fields {
			if !IsWorkgroupStorable(f.Type) {
				return false
			}
		}
		return true
	}
	return false
}

func IsTransientElement(t *Type) bool {
	return t != nil && IsHostShareable(t) && !ContainsRuntimeArray(t) && !ContainsAtomic(t)
}

// Float16Bits converts a finite value to IEEE 754 binary16 using
// round-to-nearest, ties-to-even. The boolean is false only for NaN, infinity,
// or finite values outside binary16's range.
func Float16Bits(value float64) (uint16, bool) {
	if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > 65504 {
		return 0, false
	}
	sign, magnitude := uint16(0), math.Abs(value)
	if math.Signbit(value) {
		sign = 0x8000
	}
	if magnitude < math.Ldexp(1, -14) {
		return sign | uint16(math.RoundToEven(math.Ldexp(magnitude, 24))), true
	}
	exponent := math.Ilogb(magnitude)
	significand := uint16(math.RoundToEven(math.Ldexp(magnitude, 10-exponent)))
	if significand == 2048 {
		exponent, significand = exponent+1, 1024
	}
	return sign | uint16(exponent+15)<<10 | (significand - 1024), true
}

func Float16FromBits(bits uint16) float64 {
	sign := 1.0
	if bits&0x8000 != 0 {
		sign = -1
	}
	exponent, fraction := int(bits>>10&0x1f), bits&0x03ff
	switch exponent {
	case 0:
		return sign * math.Ldexp(float64(fraction), -24)
	case 0x1f:
		if fraction == 0 {
			return math.Inf(int(sign))
		}
		return math.NaN()
	default:
		return sign * math.Ldexp(float64(1024+fraction), exponent-25)
	}
}

func ScalarLiteral(t *Type, bits uint32) string {
	switch t.Kind {
	case BoolKind:
		if bits != 0 {
			return "true"
		}
		return "false"
	case Int32Kind:
		return strconv.FormatInt(int64(int32(bits)), 10)
	case Uint32Kind:
		return strconv.FormatUint(uint64(bits), 10)
	case Float16Kind:
		return floatRaw(Float16FromBits(uint16(bits)))
	case Float32Kind:
		return floatRaw(float64(math.Float32frombits(bits)))
	default:
		panic(fmt.Sprintf("scalar raw value for %s", t))
	}
}

func floatRaw(value float64) string {
	raw := strconv.FormatFloat(value, 'g', -1, 32)
	if !strings.ContainsAny(raw, ".eE") {
		raw += ".0"
	}
	return raw
}

// TypeLayout is Tach's canonical byte layout for every host-visible type.
// Backends adapt their declarations to these compiler-owned bytes.
type TypeLayout struct {
	Size    uint32
	Align   uint32
	Stride  uint32
	Fields  []FieldLayout
	Runtime bool
}

type FieldLayout struct {
	Name   string
	Offset uint32
	Layout TypeLayout
}

func checkedLayoutSize(value uint64) (uint32, error) {
	if value > uint64(^uint32(0)) {
		return 0, fmt.Errorf("host layout exceeds the 32-bit ABI size limit")
	}
	return uint32(value), nil
}

func alignUp(alignment, value uint32) (uint32, error) {
	if alignment == 0 {
		return value, nil
	}
	return checkedLayoutSize((uint64(value) + uint64(alignment) - 1) &^ uint64(alignment-1))
}

func LayoutOf(t *Type) (TypeLayout, error) {
	return layoutOf(t, map[string]bool{})
}

func layoutOf(t *Type, seen map[string]bool) (TypeLayout, error) {
	if t == nil {
		return TypeLayout{}, fmt.Errorf("nil type")
	}
	switch t.Kind {
	case Float16Kind:
		return TypeLayout{Size: 2, Align: 2}, nil
	case Int32Kind, Uint32Kind, Float32Kind, AtomicKind:
		return TypeLayout{Size: 4, Align: 4}, nil
	case BoolKind:
		return TypeLayout{}, fmt.Errorf("bool is a value type and has no direct Tach buffer representation")
	case VectorKind:
		if !IsNumericScalar(t.Elem) {
			return TypeLayout{}, fmt.Errorf("vector element %s is not host-shareable", t.Elem)
		}
		element, err := layoutOf(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		switch t.Lanes {
		case 2:
			return TypeLayout{Size: element.Size * 2, Align: element.Align * 2}, nil
		case 3:
			return TypeLayout{Size: element.Size * 3, Align: element.Align * 4}, nil
		case 4:
			return TypeLayout{Size: element.Size * 4, Align: element.Align * 4}, nil
		default:
			return TypeLayout{}, fmt.Errorf("unsupported vector width %d", t.Lanes)
		}
	case FixedArrayKind:
		element, err := layoutOf(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		if element.Runtime || t.Count == 0 {
			return TypeLayout{}, fmt.Errorf("invalid fixed array %s", t)
		}
		stride, err := alignUp(element.Align, element.Size)
		if err != nil {
			return TypeLayout{}, err
		}
		size, err := checkedLayoutSize(uint64(stride) * uint64(t.Count))
		if err != nil {
			return TypeLayout{}, err
		}
		return TypeLayout{Size: size, Align: element.Align, Stride: stride}, nil
	case RuntimeArrayKind:
		element, err := layoutOf(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		stride, err := alignUp(element.Align, element.Size)
		if err != nil {
			return TypeLayout{}, err
		}
		return TypeLayout{Align: element.Align, Stride: stride, Runtime: true}, nil
	case StructKind:
		if seen[t.Name] {
			return TypeLayout{}, fmt.Errorf("recursive host type %s", t.Name)
		}
		seen[t.Name] = true
		defer delete(seen, t.Name)

		alignment, offset := uint32(16), uint32(0)
		fields := make([]FieldLayout, 0, len(t.Fields))
		runtimeSeen := false
		for i, field := range t.Fields {
			fieldLayout, err := layoutOf(field.Type, seen)
			if err != nil {
				return TypeLayout{}, fmt.Errorf("%s.%s: %w", t.Name, field.Name, err)
			}
			requiredAlignment := fieldLayout.Align
			if field.Type.Kind == StructKind && requiredAlignment < 16 {
				requiredAlignment = 16
			}
			offset, err = alignUp(requiredAlignment, offset)
			if err != nil {
				return TypeLayout{}, err
			}
			fields = append(fields, FieldLayout{Name: field.Name, Offset: offset, Layout: fieldLayout})
			if fieldLayout.Runtime {
				runtimeSeen = true
			} else {
				size := fieldLayout.Size
				if field.Type.Kind == StructKind {
					size, err = alignUp(16, size)
					if err != nil {
						return TypeLayout{}, err
					}
				}
				offset, err = checkedLayoutSize(uint64(offset) + uint64(size))
				if err != nil {
					return TypeLayout{}, err
				}
			}
			if requiredAlignment > alignment {
				alignment = requiredAlignment
			}
			if fieldLayout.Runtime && i != len(t.Fields)-1 {
				return TypeLayout{}, fmt.Errorf("runtime array in %s must be the final member", t.Name)
			}
		}
		if runtimeSeen {
			return TypeLayout{Size: offset, Align: alignment, Fields: fields, Runtime: true}, nil
		}
		size, err := alignUp(alignment, offset)
		if err != nil {
			return TypeLayout{}, err
		}
		return TypeLayout{Size: size, Align: alignment, Fields: fields}, nil
	case VoidKind:
		return TypeLayout{}, fmt.Errorf("void has no host layout")
	default:
		return TypeLayout{}, fmt.Errorf("unsupported host type %s", t)
	}
}

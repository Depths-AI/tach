package types

import (
	"fmt"
)

type Kind uint8

const (
	Invalid Kind = iota
	Void
	Bool
	I32
	U32
	F32
	Vector
	Struct
	Atomic
	FixedArray
	RuntimeArray
)

type Type struct {
	Kind   Kind
	Elem   *Type
	Lanes  int
	Name   string
	Fields []Field
	Count  uint32
}

type Field struct {
	Name string
	Type *Type
}

var (
	TVoid = &Type{Kind: Void, Name: "void"}
	TBool = &Type{Kind: Bool, Name: "boolean"}
	TI32  = &Type{Kind: I32, Name: "i32"}
	TU32  = &Type{Kind: U32, Name: "u32"}
	TF32  = &Type{Kind: F32, Name: "f32"}
)

func Vec(elem *Type, lanes int) *Type      { return &Type{Kind: Vector, Elem: elem, Lanes: lanes} }
func AtomicOf(elem *Type) *Type            { return &Type{Kind: Atomic, Elem: elem} }
func Array(elem *Type, count uint32) *Type { return &Type{Kind: FixedArray, Elem: elem, Count: count} }
func Runtime(elem *Type) *Type             { return &Type{Kind: RuntimeArray, Elem: elem} }

func (t *Type) String() string {
	if t == nil {
		return "<nil>"
	}
	switch t.Kind {
	case Void, Bool, I32, U32, F32:
		return t.Name
	case Vector:
		return fmt.Sprintf("%sx%d", t.Elem, t.Lanes)
	case Struct:
		return t.Name
	case Atomic:
		return fmt.Sprintf("atomic<%s>", t.Elem)
	case FixedArray:
		return fmt.Sprintf("%s[%d]", t.Elem, t.Count)
	case RuntimeArray:
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
	case Void, Bool, I32, U32, F32:
		return true
	case Vector:
		return a.Lanes == b.Lanes && Equal(a.Elem, b.Elem)
	case Struct:
		return a.Name == b.Name
	case Atomic:
		return Equal(a.Elem, b.Elem)
	case FixedArray:
		return a.Count == b.Count && Equal(a.Elem, b.Elem)
	case RuntimeArray:
		return Equal(a.Elem, b.Elem)
	}
	return false
}

func IsScalar(t *Type) bool {
	return t != nil && (t.Kind == I32 || t.Kind == U32 || t.Kind == F32 || t.Kind == Bool)
}
func IsNumericScalar(t *Type) bool {
	return t != nil && (t.Kind == I32 || t.Kind == U32 || t.Kind == F32)
}
func IsNumeric(t *Type) bool {
	return IsNumericScalar(t) || (t != nil && t.Kind == Vector && IsNumericScalar(t.Elem))
}
func IsInteger(t *Type) bool { return t != nil && (t.Kind == I32 || t.Kind == U32) }
func IsIntegerLike(t *Type) bool {
	return IsInteger(t) || t != nil && t.Kind == Vector && IsInteger(t.Elem)
}
func IsFloatLike(t *Type) bool {
	return t != nil && (t.Kind == F32 || t.Kind == Vector && t.Elem != nil && t.Elem.Kind == F32)
}
func ShiftCountType(t *Type) *Type {
	if IsInteger(t) {
		return TU32
	}
	if t != nil && t.Kind == Vector && IsInteger(t.Elem) {
		return Vec(TU32, t.Lanes)
	}
	return nil
}

// IsSignedNumeric reports whether unary negation is well-defined in Tach's
// portable numeric profile. Unsigned integers intentionally have no unary -.
func IsSignedNumeric(t *Type) bool {
	if t == nil {
		return false
	}
	if t.Kind == I32 || t.Kind == F32 {
		return true
	}
	return t.Kind == Vector && IsSignedNumeric(t.Elem)
}

// ContainsRuntimeArray is true when the type has a runtime-sized tail, directly
// or through a nested structure. Such values can be addressed in buffers but
// cannot be loaded, constructed, passed by value, or used in uniform buffers.
func ContainsRuntimeArray(t *Type) bool {
	if t == nil {
		return false
	}
	switch t.Kind {
	case RuntimeArray:
		return true
	case FixedArray:
		return ContainsRuntimeArray(t.Elem)
	case Struct:
		for _, f := range t.Fields {
			if ContainsRuntimeArray(f.Type) {
				return true
			}
		}
	}
	return false
}

// IsConstructible is the value-domain counterpart to IsHostShareable.
func IsConstructible(t *Type) bool {
	if t == nil || ContainsRuntimeArray(t) {
		return false
	}
	switch t.Kind {
	case Bool, I32, U32, F32:
		return true
	case Vector:
		return IsConstructible(t.Elem)
	case Struct:
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
	case I32, U32, F32:
		return true
	case Vector:
		return IsHostShareable(t.Elem)
	case Atomic:
		return t.Elem.Kind == I32 || t.Elem.Kind == U32
	case FixedArray:
		return false // fixed arrays are currently a workgroup-memory type in Tach
	case Struct:
		for _, f := range t.Fields {
			if !IsHostShareable(f.Type) {
				return false
			}
		}
		return true
	case RuntimeArray:
		return IsHostShareable(t.Elem)
	}
	return false
}
func FieldIndex(t *Type, name string) int {
	if t == nil || t.Kind != Struct {
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
		return TVoid
	case "boolean":
		return TBool
	case "i32":
		return TI32
	case "u32":
		return TU32
	case "f32":
		return TF32
	case "f32x2":
		return Vec(TF32, 2)
	case "f32x3":
		return Vec(TF32, 3)
	case "f32x4":
		return Vec(TF32, 4)
	case "u32x2":
		return Vec(TU32, 2)
	case "u32x3":
		return Vec(TU32, 3)
	case "u32x4":
		return Vec(TU32, 4)
	case "i32x2":
		return Vec(TI32, 2)
	case "i32x3":
		return Vec(TI32, 3)
	case "i32x4":
		return Vec(TI32, 4)
	}
	return nil
}

func ContainsAtomic(t *Type) bool {
	if t == nil {
		return false
	}
	switch t.Kind {
	case Atomic:
		return true
	case FixedArray, RuntimeArray:
		return ContainsAtomic(t.Elem)
	case Struct:
		for _, f := range t.Fields {
			if ContainsAtomic(f.Type) {
				return true
			}
		}
	}
	return false
}

func IsWorkgroupStorable(t *Type) bool {
	if t == nil || ContainsRuntimeArray(t) {
		return false
	}
	switch t.Kind {
	case I32, U32, F32:
		return true
	case Vector:
		return IsWorkgroupStorable(t.Elem)
	case Atomic:
		return t.Elem.Kind == I32 || t.Elem.Kind == U32
	case FixedArray:
		return t.Count > 0 && IsWorkgroupStorable(t.Elem)
	case Struct:
		for _, f := range t.Fields {
			if !IsWorkgroupStorable(f.Type) {
				return false
			}
		}
		return true
	}
	return false
}

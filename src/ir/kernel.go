// Package ir defines Tach's complete backend-independent program model:
// kernel computation, multi-kernel execution, and host parameter planning.
package ir

import (
	"fmt"
	"strings"

	"tach/src/foundation"
)

type ValueID uint32
type PlaceID uint32

type KernelModule struct {
	Structs   []*foundation.Type
	Functions []*Function
}

type Access uint8

const (
	Read Access = iota + 1
	Mutable
)

type BufferParam struct {
	Name   string
	Type   *foundation.Type // logical resource type; runtime array is allowed for buffers
	Access Access
	Span   foundation.Span
}

type FunctionKind uint8

const (
	Helper FunctionKind = iota + 1
	Stage
)

type SourceParamKind uint8

const (
	SourceBuffer SourceParamKind = iota + 1
	SourceValue
)

type SourceParam struct {
	Name   string
	Kind   SourceParamKind
	Buffer int
	Value  ValueID
}

type WorkgroupConstraint struct {
	Explicit bool
	Size     [3]uint32
}

type Function struct {
	Name          string
	Kind          FunctionKind
	Indices       []Param
	BufferParams  []BufferParam
	Params        []Param
	SourceParams  []SourceParam
	Return        *foundation.Type
	Body          *Block
	Workgroup     WorkgroupConstraint
	WorkgroupVars []WorkgroupVar
	Span          foundation.Span
}

type WorkgroupVar struct {
	Name string
	Type *foundation.Type
	Span foundation.Span
}

type Param struct {
	Name string
	ID   ValueID
	Type *foundation.Type
}

type Block struct {
	Instrs []Instr
	Term   Term
}

type Instr interface {
	instrNode()
	SpanOf() foundation.Span
}

type ValueDef interface {
	ResultValue() ValueID
	ResultType() *foundation.Type
}
type PlaceDef interface {
	ResultPlace() PlaceID
	PlaceType() *foundation.Type
}

type Const struct {
	Result ValueID
	Type   *foundation.Type
	Raw    string
	Span   foundation.Span
}

func (*Const) instrNode()                     {}
func (x *Const) SpanOf() foundation.Span      { return x.Span }
func (x *Const) ResultValue() ValueID         { return x.Result }
func (x *Const) ResultType() *foundation.Type { return x.Type }

// MaterializeConstant creates scalar constants and, for a vector, its final composite.
func MaterializeConstant(value *foundation.ConstantValue, span foundation.Span, fresh func() ValueID) (ValueID, []Instr) {
	element := value.Type
	if value.Type.Kind == foundation.VectorKind {
		element = value.Type.Elem
	}
	components := make([]ValueID, len(value.Bits))
	instructions := make([]Instr, 0, len(value.Bits)+1)
	for index, bits := range value.Bits {
		components[index] = fresh()
		instructions = append(instructions, &Const{Result: components[index], Type: element, Raw: foundation.ScalarLiteral(element, bits), Span: span})
	}
	if value.Type.Kind != foundation.VectorKind {
		return components[0], instructions
	}
	result := fresh()
	return result, append(instructions, &Composite{Result: result, Type: value.Type, Values: components, Span: span})
}

type Unary struct {
	Result ValueID
	Type   *foundation.Type
	Op     string
	X      ValueID
	Span   foundation.Span
}

func (*Unary) instrNode()                     {}
func (x *Unary) SpanOf() foundation.Span      { return x.Span }
func (x *Unary) ResultValue() ValueID         { return x.Result }
func (x *Unary) ResultType() *foundation.Type { return x.Type }

type Binary struct {
	Result      ValueID
	Type        *foundation.Type
	Op          string
	Left, Right ValueID
	Span        foundation.Span
}

func (*Binary) instrNode()                     {}
func (x *Binary) SpanOf() foundation.Span      { return x.Span }
func (x *Binary) ResultValue() ValueID         { return x.Result }
func (x *Binary) ResultType() *foundation.Type { return x.Type }

type Convert struct {
	Result ValueID
	Type   *foundation.Type
	X      ValueID
	From   *foundation.Type
	Span   foundation.Span
}

func (*Convert) instrNode()                     {}
func (x *Convert) SpanOf() foundation.Span      { return x.Span }
func (x *Convert) ResultValue() ValueID         { return x.Result }
func (x *Convert) ResultType() *foundation.Type { return x.Type }

type Composite struct {
	Result ValueID
	Type   *foundation.Type
	Values []ValueID
	Span   foundation.Span
}

func (*Composite) instrNode()                     {}
func (x *Composite) SpanOf() foundation.Span      { return x.Span }
func (x *Composite) ResultValue() ValueID         { return x.Result }
func (x *Composite) ResultType() *foundation.Type { return x.Type }

type Extract struct {
	Result ValueID
	Type   *foundation.Type
	Base   ValueID
	Index  int
	Span   foundation.Span
}

func (*Extract) instrNode()                     {}
func (x *Extract) SpanOf() foundation.Span      { return x.Span }
func (x *Extract) ResultValue() ValueID         { return x.Result }
func (x *Extract) ResultType() *foundation.Type { return x.Type }

type VectorIndex struct {
	Result ValueID
	Type   *foundation.Type
	Base   ValueID
	Index  ValueID
	Span   foundation.Span
}

func (*VectorIndex) instrNode()                     {}
func (x *VectorIndex) SpanOf() foundation.Span      { return x.Span }
func (x *VectorIndex) ResultValue() ValueID         { return x.Result }
func (x *VectorIndex) ResultType() *foundation.Type { return x.Type }

type Call struct {
	Result   ValueID
	Type     *foundation.Type
	Function string
	Args     []ValueID
	Span     foundation.Span
}

func (*Call) instrNode()                     {}
func (x *Call) SpanOf() foundation.Span      { return x.Span }
func (x *Call) ResultValue() ValueID         { return x.Result }
func (x *Call) ResultType() *foundation.Type { return x.Type }

type IntrinsicKind uint8

const (
	IntrinsicAbs IntrinsicKind = iota + 1
	IntrinsicFloor
	IntrinsicCeil
	IntrinsicTrunc
	IntrinsicSin
	IntrinsicCos
	IntrinsicTan
	IntrinsicExp
	IntrinsicExp2
	IntrinsicLog
	IntrinsicLog2
	IntrinsicSqrt
	IntrinsicRSqrt
	IntrinsicPow
	IntrinsicMin
	IntrinsicMax
	IntrinsicClamp
	IntrinsicFma
	IntrinsicDot
	IntrinsicLength
	IntrinsicDistance
	IntrinsicCross
	IntrinsicNormalize
	IntrinsicAll
	IntrinsicAny
	IntrinsicSelect
)

type NumericDomain uint8

const (
	NumericAny NumericDomain = iota
	NumericSigned
	NumericFloat
)

type IntrinsicRule struct {
	Arity         int
	Domain        NumericDomain
	Broadcast     uint8
	VectorOnly    bool
	ResultElement bool
	Lanes         int
}

func (k IntrinsicKind) Rule() IntrinsicRule {
	switch k {
	case IntrinsicAbs:
		return IntrinsicRule{Arity: 1, Domain: NumericSigned}
	case IntrinsicFloor, IntrinsicCeil, IntrinsicTrunc, IntrinsicSin, IntrinsicCos, IntrinsicTan, IntrinsicExp, IntrinsicExp2, IntrinsicLog, IntrinsicLog2, IntrinsicSqrt, IntrinsicRSqrt:
		return IntrinsicRule{Arity: 1, Domain: NumericFloat}
	case IntrinsicPow:
		return IntrinsicRule{Arity: 2, Domain: NumericFloat, Broadcast: 1 << 1}
	case IntrinsicMin, IntrinsicMax:
		return IntrinsicRule{Arity: 2, Domain: NumericAny, Broadcast: 0b11}
	case IntrinsicClamp:
		return IntrinsicRule{Arity: 3, Domain: NumericAny, Broadcast: 0b111}
	case IntrinsicFma:
		return IntrinsicRule{Arity: 3, Domain: NumericFloat, Broadcast: 0b111}
	case IntrinsicDot, IntrinsicDistance:
		return IntrinsicRule{Arity: 2, Domain: NumericFloat, VectorOnly: true, ResultElement: true}
	case IntrinsicLength:
		return IntrinsicRule{Arity: 1, Domain: NumericFloat, VectorOnly: true, ResultElement: true}
	case IntrinsicCross:
		return IntrinsicRule{Arity: 2, Domain: NumericFloat, VectorOnly: true, Lanes: 3}
	case IntrinsicNormalize:
		return IntrinsicRule{Arity: 1, Domain: NumericFloat, VectorOnly: true}
	default:
		return IntrinsicRule{}
	}
}

func (d NumericDomain) Accepts(t *foundation.Type) bool {
	switch d {
	case NumericAny:
		return foundation.IsNumericScalar(t)
	case NumericSigned:
		return t != nil && (t.Kind == foundation.Int32Kind || t.Kind == foundation.Float16Kind || t.Kind == foundation.Float32Kind)
	case NumericFloat:
		return t != nil && (t.Kind == foundation.Float16Kind || t.Kind == foundation.Float32Kind)
	default:
		return false
	}
}

func (d NumericDomain) String() string {
	switch d {
	case NumericAny:
		return "numeric"
	case NumericSigned:
		return "signed numeric"
	case NumericFloat:
		return "floating-point"
	default:
		return "invalid"
	}
}

type Intrinsic struct {
	Result ValueID
	Type   *foundation.Type
	Kind   IntrinsicKind
	Args   []ValueID
	Span   foundation.Span
}

func (*Intrinsic) instrNode()                     {}
func (x *Intrinsic) SpanOf() foundation.Span      { return x.Span }
func (x *Intrinsic) ResultValue() ValueID         { return x.Result }
func (x *Intrinsic) ResultType() *foundation.Type { return x.Type }

type PlaceRoot struct {
	Result PlaceID
	Type   *foundation.Type
	Buffer int
	Span   foundation.Span
}

func (*PlaceRoot) instrNode()                    {}
func (x *PlaceRoot) SpanOf() foundation.Span     { return x.Span }
func (x *PlaceRoot) ResultPlace() PlaceID        { return x.Result }
func (x *PlaceRoot) PlaceType() *foundation.Type { return x.Type }

type PlaceWorkgroup struct {
	Result    PlaceID
	Type      *foundation.Type
	Workgroup int
	Span      foundation.Span
}

func (*PlaceWorkgroup) instrNode()                    {}
func (x *PlaceWorkgroup) SpanOf() foundation.Span     { return x.Span }
func (x *PlaceWorkgroup) ResultPlace() PlaceID        { return x.Result }
func (x *PlaceWorkgroup) PlaceType() *foundation.Type { return x.Type }

type PlaceField struct {
	Result PlaceID
	Type   *foundation.Type
	Base   PlaceID
	Field  int
	Span   foundation.Span
}

func (*PlaceField) instrNode()                    {}
func (x *PlaceField) SpanOf() foundation.Span     { return x.Span }
func (x *PlaceField) ResultPlace() PlaceID        { return x.Result }
func (x *PlaceField) PlaceType() *foundation.Type { return x.Type }

type PlaceIndex struct {
	Result PlaceID
	Type   *foundation.Type
	Base   PlaceID
	Index  ValueID
	Span   foundation.Span
}

func (*PlaceIndex) instrNode()                    {}
func (x *PlaceIndex) SpanOf() foundation.Span     { return x.Span }
func (x *PlaceIndex) ResultPlace() PlaceID        { return x.Result }
func (x *PlaceIndex) PlaceType() *foundation.Type { return x.Type }

type Load struct {
	Result ValueID
	Type   *foundation.Type
	Place  PlaceID
	Span   foundation.Span
}

func (*Load) instrNode()                     {}
func (x *Load) SpanOf() foundation.Span      { return x.Span }
func (x *Load) ResultValue() ValueID         { return x.Result }
func (x *Load) ResultType() *foundation.Type { return x.Type }

type Store struct {
	Place PlaceID
	Value ValueID
	Span  foundation.Span
}

func (*Store) instrNode()                {}
func (x *Store) SpanOf() foundation.Span { return x.Span }

type ArrayLength struct {
	Result ValueID
	Type   *foundation.Type
	Place  PlaceID
	Span   foundation.Span
}

func (*ArrayLength) instrNode()                     {}
func (x *ArrayLength) SpanOf() foundation.Span      { return x.Span }
func (x *ArrayLength) ResultValue() ValueID         { return x.Result }
func (x *ArrayLength) ResultType() *foundation.Type { return x.Type }

type If struct {
	Results    []Result
	Cond       ValueID
	Then, Else *Block
	Span       foundation.Span
}

func (*If) instrNode()                {}
func (x *If) SpanOf() foundation.Span { return x.Span }

type Loop struct {
	Results []Result
	Params  []LoopParam
	Cond    *Block
	Body    *Block
	Span    foundation.Span
}

func (*Loop) instrNode()                {}
func (x *Loop) SpanOf() foundation.Span { return x.Span }

type Scope struct {
	Body *Block
	Span foundation.Span
}

func (*Scope) instrNode()                {}
func (x *Scope) SpanOf() foundation.Span { return x.Span }

type AtomicKind uint8

const (
	AtomicLoad AtomicKind = iota + 1
	AtomicStore
	AtomicAdd
	AtomicSub
	AtomicMin
	AtomicMax
	AtomicAnd
	AtomicOr
	AtomicXor
	AtomicExchange
	AtomicCompareExchange
)

type Atomic struct {
	Result   ValueID          // zero only for AtomicStore
	Type     *foundation.Type // underlying int32/uint32 result/value type
	Op       AtomicKind
	Place    PlaceID
	Value    ValueID // zero only for AtomicLoad
	Expected ValueID // nonzero only for AtomicCompareExchange
	Span     foundation.Span
}

func (*Atomic) instrNode()                     {}
func (x *Atomic) SpanOf() foundation.Span      { return x.Span }
func (x *Atomic) ResultValue() ValueID         { return x.Result }
func (x *Atomic) ResultType() *foundation.Type { return x.Type }

type BarrierKind uint8

const (
	BarrierWorkgroup BarrierKind = iota + 1
	BarrierBuffer
)

type Barrier struct {
	Kind BarrierKind
	Span foundation.Span
}

func (*Barrier) instrNode()                {}
func (x *Barrier) SpanOf() foundation.Span { return x.Span }

type Result struct {
	ID   ValueID
	Type *foundation.Type
}
type LoopParam struct {
	ID   ValueID
	Type *foundation.Type
	Init ValueID
}

type Term interface{ termNode() }
type Yield struct{ Values []ValueID }

func (*Yield) termNode() {}

type Continue struct{ Values []ValueID }

func (*Continue) termNode() {}

type Break struct{ Values []ValueID }

func (*Break) termNode() {}

type Return struct {
	Value    ValueID
	HasValue bool
}

func (*Return) termNode() {}

type Unreachable struct{}

func (*Unreachable) termNode() {}

type ExitScope struct{}

func (*ExitScope) termNode() {}

func (m *KernelModule) Function(name string) *Function {
	for _, f := range m.Functions {
		if f.Name == name {
			return f
		}
	}
	return nil
}

func CloneKernel(m *KernelModule) *KernelModule {
	if m == nil {
		return nil
	}
	out := &KernelModule{Structs: append([]*foundation.Type(nil), m.Structs...), Functions: make([]*Function, len(m.Functions))}
	for i, f := range m.Functions {
		g := *f
		g.Indices = append([]Param(nil), f.Indices...)
		g.BufferParams = append([]BufferParam(nil), f.BufferParams...)
		g.Params = append([]Param(nil), f.Params...)
		g.SourceParams = append([]SourceParam(nil), f.SourceParams...)
		g.WorkgroupVars = append([]WorkgroupVar(nil), f.WorkgroupVars...)
		g.Body = cloneBlock(f.Body)
		out.Functions[i] = &g
	}
	return out
}

func cloneBlock(block *Block) *Block {
	if block == nil {
		return nil
	}
	out := &Block{Instrs: make([]Instr, len(block.Instrs)), Term: cloneTerm(block.Term)}
	for i, instruction := range block.Instrs {
		out.Instrs[i] = cloneInstr(instruction)
	}
	return out
}

func cloneInstr(instruction Instr) Instr {
	switch x := instruction.(type) {
	case *Const:
		y := *x
		return &y
	case *Unary:
		y := *x
		return &y
	case *Binary:
		y := *x
		return &y
	case *Convert:
		y := *x
		return &y
	case *Composite:
		y := *x
		y.Values = append([]ValueID(nil), x.Values...)
		return &y
	case *Extract:
		y := *x
		return &y
	case *VectorIndex:
		y := *x
		return &y
	case *Call:
		y := *x
		y.Args = append([]ValueID(nil), x.Args...)
		return &y
	case *Intrinsic:
		y := *x
		y.Args = append([]ValueID(nil), x.Args...)
		return &y
	case *PlaceRoot:
		y := *x
		return &y
	case *PlaceWorkgroup:
		y := *x
		return &y
	case *PlaceField:
		y := *x
		return &y
	case *PlaceIndex:
		y := *x
		return &y
	case *Load:
		y := *x
		return &y
	case *Store:
		y := *x
		return &y
	case *ArrayLength:
		y := *x
		return &y
	case *If:
		y := *x
		y.Results = append([]Result(nil), x.Results...)
		y.Then = cloneBlock(x.Then)
		y.Else = cloneBlock(x.Else)
		return &y
	case *Loop:
		y := *x
		y.Results = append([]Result(nil), x.Results...)
		y.Params = append([]LoopParam(nil), x.Params...)
		y.Cond = cloneBlock(x.Cond)
		y.Body = cloneBlock(x.Body)
		return &y
	case *Scope:
		y := *x
		y.Body = cloneBlock(x.Body)
		return &y
	case *Atomic:
		y := *x
		return &y
	case *Barrier:
		y := *x
		return &y
	default:
		panic(fmt.Sprintf("clone unknown instruction %T", instruction))
	}
}

func cloneTerm(term Term) Term {
	switch x := term.(type) {
	case nil:
		return nil
	case *Yield:
		y := *x
		y.Values = append([]ValueID(nil), x.Values...)
		return &y
	case *Continue:
		y := *x
		y.Values = append([]ValueID(nil), x.Values...)
		return &y
	case *Break:
		y := *x
		y.Values = append([]ValueID(nil), x.Values...)
		return &y
	case *Return:
		y := *x
		return &y
	case *Unreachable:
		return &Unreachable{}
	case *ExitScope:
		return &ExitScope{}
	default:
		panic(fmt.Sprintf("clone unknown terminator %T", term))
	}
}

func DumpKernel(m *KernelModule) string {
	var b strings.Builder
	for _, s := range m.Structs {
		fmt.Fprintf(&b, "type %s = {\n", s.Name)
		for _, f := range s.Fields {
			fmt.Fprintf(&b, "  %s: %s\n", f.Name, f.Type)
		}
		b.WriteString("}\n\n")
	}
	for _, f := range m.Functions {
		dumpFunc(&b, f)
		b.WriteByte('\n')
	}
	return b.String()
}
func (a Access) String() string {
	switch a {
	case Read:
		return "read"
	case Mutable:
		return "mutable"
	default:
		return fmt.Sprintf("access(%d)", a)
	}
}
func (k IntrinsicKind) String() string {
	switch k {
	case IntrinsicAbs:
		return "abs"
	case IntrinsicFloor:
		return "floor"
	case IntrinsicCeil:
		return "ceil"
	case IntrinsicTrunc:
		return "trunc"
	case IntrinsicSin:
		return "sin"
	case IntrinsicCos:
		return "cos"
	case IntrinsicTan:
		return "tan"
	case IntrinsicExp:
		return "exp"
	case IntrinsicExp2:
		return "exp2"
	case IntrinsicLog:
		return "log"
	case IntrinsicLog2:
		return "log2"
	case IntrinsicSqrt:
		return "sqrt"
	case IntrinsicRSqrt:
		return "rsqrt"
	case IntrinsicPow:
		return "pow"
	case IntrinsicMin:
		return "min"
	case IntrinsicMax:
		return "max"
	case IntrinsicClamp:
		return "clamp"
	case IntrinsicFma:
		return "fma"
	case IntrinsicDot:
		return "dot"
	case IntrinsicLength:
		return "length"
	case IntrinsicDistance:
		return "distance"
	case IntrinsicCross:
		return "cross"
	case IntrinsicNormalize:
		return "normalize"
	case IntrinsicAll:
		return "all"
	case IntrinsicAny:
		return "any"
	case IntrinsicSelect:
		return "select"
	default:
		return fmt.Sprintf("intrinsic(%d)", k)
	}
}

func (k AtomicKind) String() string {
	switch k {
	case AtomicLoad:
		return "atomic_load"
	case AtomicStore:
		return "atomic_store"
	case AtomicAdd:
		return "atomic_add"
	case AtomicSub:
		return "atomic_sub"
	case AtomicMin:
		return "atomic_min"
	case AtomicMax:
		return "atomic_max"
	case AtomicAnd:
		return "atomic_and"
	case AtomicOr:
		return "atomic_or"
	case AtomicXor:
		return "atomic_xor"
	case AtomicExchange:
		return "atomic_exchange"
	case AtomicCompareExchange:
		return "atomic_compare_exchange"
	default:
		return fmt.Sprintf("atomic(%d)", k)
	}
}
func (k BarrierKind) String() string {
	switch k {
	case BarrierWorkgroup:
		return "workgroup_barrier"
	case BarrierBuffer:
		return "buffer_barrier"
	default:
		return fmt.Sprintf("barrier(%d)", k)
	}
}
func dumpFunc(b *strings.Builder, f *Function) {
	parameterTypes := make(map[ValueID]*foundation.Type, len(f.Params))
	for _, parameter := range f.Params {
		parameterTypes[parameter.ID] = parameter.Type
	}
	if f.Kind == Stage {
		fmt.Fprintf(b, "stage @%s[", f.Name)
		for i, index := range f.Indices {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(b, "%s=%%%d", index.Name, index.ID)
		}
		b.WriteString("](")
		for i, p := range f.SourceParams {
			if i > 0 {
				b.WriteString(", ")
			}
			if p.Kind == SourceValue {
				fmt.Fprintf(b, "%s=%%%d: %s", p.Name, p.Value, parameterTypes[p.Value])
			} else {
				fmt.Fprintf(b, "%s=%%b%d: %s access=%s", p.Name, p.Buffer, f.BufferParams[p.Buffer].Type, f.BufferParams[p.Buffer].Access)
			}
		}
		if f.Workgroup.Explicit {
			fmt.Fprintf(b, ") workgroup(%d,%d,%d) {\n", f.Workgroup.Size[0], f.Workgroup.Size[1], f.Workgroup.Size[2])
		} else {
			b.WriteString(") workgroup(auto) {\n")
		}
		for i, w := range f.WorkgroupVars {
			fmt.Fprintf(b, "  workgroup @%d %s: %s\n", i, w.Name, w.Type)
		}
	} else {
		fmt.Fprintf(b, "function @%s(", f.Name)
		for i, p := range f.Params {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(b, "%%%d: %s", p.ID, p.Type)
		}
		fmt.Fprintf(b, ") -> %s {\n", f.Return)
	}
	dumpBlock(b, f.Body, "  ")
	b.WriteString("}\n")
}
func dumpBlock(b *strings.Builder, bl *Block, ind string) {
	for _, in := range bl.Instrs {
		switch x := in.(type) {
		case *Const:
			fmt.Fprintf(b, "%s%%%d = const %s %s\n", ind, x.Result, x.Type, x.Raw)
		case *Unary:
			fmt.Fprintf(b, "%s%%%d = %s %%%d : %s\n", ind, x.Result, x.Op, x.X, x.Type)
		case *Binary:
			fmt.Fprintf(b, "%s%%%d = %s %%%d, %%%d : %s\n", ind, x.Result, x.Op, x.Left, x.Right, x.Type)
		case *Convert:
			fmt.Fprintf(b, "%s%%%d = convert %%%d : %s -> %s\n", ind, x.Result, x.X, x.From, x.Type)
		case *Composite:
			fmt.Fprintf(b, "%s%%%d = composite %s %v\n", ind, x.Result, x.Type, x.Values)
		case *Extract:
			fmt.Fprintf(b, "%s%%%d = extract %%%d[%d] : %s\n", ind, x.Result, x.Base, x.Index, x.Type)
		case *VectorIndex:
			fmt.Fprintf(b, "%s%%%d = vector_index %%%d, %%%d : %s\n", ind, x.Result, x.Base, x.Index, x.Type)
		case *Call:
			fmt.Fprintf(b, "%s%%%d = call @%s%v : %s\n", ind, x.Result, x.Function, x.Args, x.Type)
		case *Intrinsic:
			fmt.Fprintf(b, "%s%%%d = intrinsic %s%v : %s\n", ind, x.Result, x.Kind, x.Args, x.Type)
		case *PlaceRoot:
			fmt.Fprintf(b, "%s&p%d = place.buffer %%b%d : %s\n", ind, x.Result, x.Buffer, x.Type)
		case *PlaceWorkgroup:
			fmt.Fprintf(b, "%s&p%d = place.workgroup @%d : %s\n", ind, x.Result, x.Workgroup, x.Type)
		case *PlaceField:
			fmt.Fprintf(b, "%s&p%d = place.field &p%d[%d] : %s\n", ind, x.Result, x.Base, x.Field, x.Type)
		case *PlaceIndex:
			fmt.Fprintf(b, "%s&p%d = place.index &p%d, %%%d : %s\n", ind, x.Result, x.Base, x.Index, x.Type)
		case *Load:
			fmt.Fprintf(b, "%s%%%d = load &p%d : %s\n", ind, x.Result, x.Place, x.Type)
		case *Store:
			fmt.Fprintf(b, "%sstore &p%d, %%%d\n", ind, x.Place, x.Value)
		case *ArrayLength:
			fmt.Fprintf(b, "%s%%%d = array_length &p%d\n", ind, x.Result, x.Place)
		case *Atomic:
			if x.Op == AtomicLoad {
				fmt.Fprintf(b, "%s%%%d = %s &p%d : %s\n", ind, x.Result, x.Op, x.Place, x.Type)
			} else if x.Op == AtomicStore {
				fmt.Fprintf(b, "%s%s &p%d, %%%d\n", ind, x.Op, x.Place, x.Value)
			} else if x.Op == AtomicCompareExchange {
				fmt.Fprintf(b, "%s%%%d = %s &p%d, %%%d, %%%d : %s\n", ind, x.Result, x.Op, x.Place, x.Expected, x.Value, x.Type)
			} else {
				fmt.Fprintf(b, "%s%%%d = %s &p%d, %%%d : %s\n", ind, x.Result, x.Op, x.Place, x.Value, x.Type)
			}
		case *Barrier:
			fmt.Fprintf(b, "%s%s\n", ind, x.Kind)
		case *If:
			fmt.Fprintf(b, "%sif %%%d -> %v {\n", ind, x.Cond, x.Results)
			dumpBlock(b, x.Then, ind+"  ")
			fmt.Fprintf(b, "%s} else {\n", ind)
			dumpBlock(b, x.Else, ind+"  ")
			fmt.Fprintf(b, "%s}\n", ind)
		case *Loop:
			fmt.Fprintf(b, "%sloop params=%v -> %v { cond\n", ind, x.Params, x.Results)
			dumpBlock(b, x.Cond, ind+"  ")
			fmt.Fprintf(b, "%sbody\n", ind)
			dumpBlock(b, x.Body, ind+"  ")
			fmt.Fprintf(b, "%s}\n", ind)
		case *Scope:
			fmt.Fprintf(b, "%sscope {\n", ind)
			dumpBlock(b, x.Body, ind+"  ")
			fmt.Fprintf(b, "%s}\n", ind)
		}
	}
	switch t := bl.Term.(type) {
	case *Yield:
		fmt.Fprintf(b, "%syield %v\n", ind, t.Values)
	case *Continue:
		fmt.Fprintf(b, "%scontinue %v\n", ind, t.Values)
	case *Break:
		fmt.Fprintf(b, "%sbreak %v\n", ind, t.Values)
	case *Return:
		if t.HasValue {
			fmt.Fprintf(b, "%sreturn %%%d\n", ind, t.Value)
		} else {
			fmt.Fprintf(b, "%sreturn\n", ind)
		}
	case *Unreachable:
		fmt.Fprintf(b, "%sunreachable\n", ind)
	case *ExitScope:
		fmt.Fprintf(b, "%sexit_scope\n", ind)
	}
}

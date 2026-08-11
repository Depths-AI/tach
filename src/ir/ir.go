package ir

import (
	"fmt"
	"strings"

	"tach/src/source"
	"tach/src/types"
)

type ValueID uint32
type PlaceID uint32

type Module struct {
	Structs   []*types.Type
	Resources []Resource
	Functions []*Function
}

type Access uint8

const (
	Read Access = iota + 1
	Mutable
)

type Resource struct {
	Name   string
	Type   *types.Type // logical resource type; runtime array is allowed for buffers
	Access Access
	Span   source.Span
}

type Function struct {
	Name          string
	Indices       []Param
	Params        []Param
	Return        *types.Type
	Body          *Block
	Compute       bool
	Workgroup     [3]uint32
	KernelParams  []KernelParam
	WorkgroupVars []WorkgroupVar
	Span          source.Span
}

type WorkgroupVar struct {
	Name string
	Type *types.Type
	Span source.Span
}

type KernelParam struct {
	Name     string
	Value    ValueID // zero means this parameter names a buffer resource
	Resource int
}

type Param struct {
	Name string
	ID   ValueID
	Type *types.Type
}

type Block struct {
	Instrs []Instr
	Term   Term
}

type Instr interface {
	instrNode()
	SpanOf() source.Span
}

type ValueDef interface {
	ResultValue() ValueID
	ResultType() *types.Type
}
type PlaceDef interface {
	ResultPlace() PlaceID
	PlaceType() *types.Type
}

type Const struct {
	Result ValueID
	Type   *types.Type
	Raw    string
	Span   source.Span
}

func (*Const) instrNode()                {}
func (x *Const) SpanOf() source.Span     { return x.Span }
func (x *Const) ResultValue() ValueID    { return x.Result }
func (x *Const) ResultType() *types.Type { return x.Type }

type Unary struct {
	Result ValueID
	Type   *types.Type
	Op     string
	X      ValueID
	Span   source.Span
}

func (*Unary) instrNode()                {}
func (x *Unary) SpanOf() source.Span     { return x.Span }
func (x *Unary) ResultValue() ValueID    { return x.Result }
func (x *Unary) ResultType() *types.Type { return x.Type }

type Binary struct {
	Result      ValueID
	Type        *types.Type
	Op          string
	Left, Right ValueID
	Span        source.Span
}

func (*Binary) instrNode()                {}
func (x *Binary) SpanOf() source.Span     { return x.Span }
func (x *Binary) ResultValue() ValueID    { return x.Result }
func (x *Binary) ResultType() *types.Type { return x.Type }

type Convert struct {
	Result ValueID
	Type   *types.Type
	X      ValueID
	From   *types.Type
	Span   source.Span
}

func (*Convert) instrNode()                {}
func (x *Convert) SpanOf() source.Span     { return x.Span }
func (x *Convert) ResultValue() ValueID    { return x.Result }
func (x *Convert) ResultType() *types.Type { return x.Type }

type Composite struct {
	Result ValueID
	Type   *types.Type
	Values []ValueID
	Span   source.Span
}

func (*Composite) instrNode()                {}
func (x *Composite) SpanOf() source.Span     { return x.Span }
func (x *Composite) ResultValue() ValueID    { return x.Result }
func (x *Composite) ResultType() *types.Type { return x.Type }

type Extract struct {
	Result ValueID
	Type   *types.Type
	Base   ValueID
	Index  int
	Span   source.Span
}

func (*Extract) instrNode()                {}
func (x *Extract) SpanOf() source.Span     { return x.Span }
func (x *Extract) ResultValue() ValueID    { return x.Result }
func (x *Extract) ResultType() *types.Type { return x.Type }

type VectorIndex struct {
	Result ValueID
	Type   *types.Type
	Base   ValueID
	Index  ValueID
	Span   source.Span
}

func (*VectorIndex) instrNode()                {}
func (x *VectorIndex) SpanOf() source.Span     { return x.Span }
func (x *VectorIndex) ResultValue() ValueID    { return x.Result }
func (x *VectorIndex) ResultType() *types.Type { return x.Type }

type Call struct {
	Result   ValueID
	Type     *types.Type
	Function string
	Args     []ValueID
	Span     source.Span
}

func (*Call) instrNode()                {}
func (x *Call) SpanOf() source.Span     { return x.Span }
func (x *Call) ResultValue() ValueID    { return x.Result }
func (x *Call) ResultType() *types.Type { return x.Type }

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
	IntrinsicDot
	IntrinsicLength
	IntrinsicDistance
	IntrinsicCross
	IntrinsicNormalize
)

type Intrinsic struct {
	Result ValueID
	Type   *types.Type
	Kind   IntrinsicKind
	Args   []ValueID
	Span   source.Span
}

func (*Intrinsic) instrNode()                {}
func (x *Intrinsic) SpanOf() source.Span     { return x.Span }
func (x *Intrinsic) ResultValue() ValueID    { return x.Result }
func (x *Intrinsic) ResultType() *types.Type { return x.Type }

type PlaceRoot struct {
	Result   PlaceID
	Type     *types.Type
	Resource int
	Span     source.Span
}

func (*PlaceRoot) instrNode()               {}
func (x *PlaceRoot) SpanOf() source.Span    { return x.Span }
func (x *PlaceRoot) ResultPlace() PlaceID   { return x.Result }
func (x *PlaceRoot) PlaceType() *types.Type { return x.Type }

type PlaceWorkgroup struct {
	Result    PlaceID
	Type      *types.Type
	Workgroup int
	Span      source.Span
}

func (*PlaceWorkgroup) instrNode()               {}
func (x *PlaceWorkgroup) SpanOf() source.Span    { return x.Span }
func (x *PlaceWorkgroup) ResultPlace() PlaceID   { return x.Result }
func (x *PlaceWorkgroup) PlaceType() *types.Type { return x.Type }

type PlaceField struct {
	Result PlaceID
	Type   *types.Type
	Base   PlaceID
	Field  int
	Span   source.Span
}

func (*PlaceField) instrNode()               {}
func (x *PlaceField) SpanOf() source.Span    { return x.Span }
func (x *PlaceField) ResultPlace() PlaceID   { return x.Result }
func (x *PlaceField) PlaceType() *types.Type { return x.Type }

type PlaceIndex struct {
	Result PlaceID
	Type   *types.Type
	Base   PlaceID
	Index  ValueID
	Span   source.Span
}

func (*PlaceIndex) instrNode()               {}
func (x *PlaceIndex) SpanOf() source.Span    { return x.Span }
func (x *PlaceIndex) ResultPlace() PlaceID   { return x.Result }
func (x *PlaceIndex) PlaceType() *types.Type { return x.Type }

type Load struct {
	Result ValueID
	Type   *types.Type
	Place  PlaceID
	Span   source.Span
}

func (*Load) instrNode()                {}
func (x *Load) SpanOf() source.Span     { return x.Span }
func (x *Load) ResultValue() ValueID    { return x.Result }
func (x *Load) ResultType() *types.Type { return x.Type }

type Store struct {
	Place PlaceID
	Value ValueID
	Span  source.Span
}

func (*Store) instrNode()            {}
func (x *Store) SpanOf() source.Span { return x.Span }

type ArrayLength struct {
	Result ValueID
	Type   *types.Type
	Place  PlaceID
	Span   source.Span
}

func (*ArrayLength) instrNode()                {}
func (x *ArrayLength) SpanOf() source.Span     { return x.Span }
func (x *ArrayLength) ResultValue() ValueID    { return x.Result }
func (x *ArrayLength) ResultType() *types.Type { return x.Type }

type If struct {
	Results    []Result
	Cond       ValueID
	Then, Else *Block
	Span       source.Span
}

func (*If) instrNode()            {}
func (x *If) SpanOf() source.Span { return x.Span }

type Loop struct {
	Results []Result
	Params  []LoopParam
	Cond    *Block
	Body    *Block
	Span    source.Span
}

func (*Loop) instrNode()            {}
func (x *Loop) SpanOf() source.Span { return x.Span }

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
)

type Atomic struct {
	Result ValueID     // zero only for AtomicStore
	Type   *types.Type // underlying int32/uint32 result/value type
	Op     AtomicKind
	Place  PlaceID
	Value  ValueID // zero only for AtomicLoad
	Span   source.Span
}

func (*Atomic) instrNode()                {}
func (x *Atomic) SpanOf() source.Span     { return x.Span }
func (x *Atomic) ResultValue() ValueID    { return x.Result }
func (x *Atomic) ResultType() *types.Type { return x.Type }

type BarrierKind uint8

const (
	BarrierWorkgroup BarrierKind = iota + 1
	BarrierBuffer
)

type Barrier struct {
	Kind BarrierKind
	Span source.Span
}

func (*Barrier) instrNode()            {}
func (x *Barrier) SpanOf() source.Span { return x.Span }

type Result struct {
	ID   ValueID
	Type *types.Type
}
type LoopParam struct {
	ID   ValueID
	Type *types.Type
	Init ValueID
}

type Term interface{ termNode() }
type Yield struct{ Values []ValueID }

func (*Yield) termNode() {}

type Continue struct{ Values []ValueID }

func (*Continue) termNode() {}

type Return struct {
	Value    ValueID
	HasValue bool
}

func (*Return) termNode() {}

type Unreachable struct{}

func (*Unreachable) termNode() {}

func (m *Module) Function(name string) *Function {
	for _, f := range m.Functions {
		if f.Name == name {
			return f
		}
	}
	return nil
}

func Dump(m *Module) string {
	var b strings.Builder
	for _, s := range m.Structs {
		fmt.Fprintf(&b, "type %s = {\n", s.Name)
		for _, f := range s.Fields {
			fmt.Fprintf(&b, "  %s: %s\n", f.Name, f.Type)
		}
		b.WriteString("}\n\n")
	}
	for i, r := range m.Resources {
		fmt.Fprintf(&b, "buffer @%d %s: %s access=%s\n", i, r.Name, r.Type, r.Access)
	}
	if len(m.Resources) > 0 {
		b.WriteByte('\n')
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
	parameterTypes := make(map[ValueID]*types.Type, len(f.Params))
	for _, parameter := range f.Params {
		parameterTypes[parameter.ID] = parameter.Type
	}
	if f.Compute {
		fmt.Fprintf(b, "compute @%s[", f.Name)
		for i, index := range f.Indices {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(b, "%s=%%%d", index.Name, index.ID)
		}
		b.WriteString("](")
		for i, p := range f.KernelParams {
			if i > 0 {
				b.WriteString(", ")
			}
			if p.Value != 0 {
				fmt.Fprintf(b, "%s=%%%d: %s", p.Name, p.Value, parameterTypes[p.Value])
			} else {
				fmt.Fprintf(b, "%s=@%d", p.Name, p.Resource)
			}
		}
		fmt.Fprintf(b, ") workgroup(%d,%d,%d) {\n", f.Workgroup[0], f.Workgroup[1], f.Workgroup[2])
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
			fmt.Fprintf(b, "%s&p%d = place.resource @%d : %s\n", ind, x.Result, x.Resource, x.Type)
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
		}
	}
	switch t := bl.Term.(type) {
	case *Yield:
		fmt.Fprintf(b, "%syield %v\n", ind, t.Values)
	case *Continue:
		fmt.Fprintf(b, "%scontinue %v\n", ind, t.Values)
	case *Return:
		if t.HasValue {
			fmt.Fprintf(b, "%sreturn %%%d\n", ind, t.Value)
		} else {
			fmt.Fprintf(b, "%sreturn\n", ind)
		}
	case *Unreachable:
		fmt.Fprintf(b, "%sunreachable\n", ind)
	}
}

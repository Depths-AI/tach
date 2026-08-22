package ir

import "tach/foundation"

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

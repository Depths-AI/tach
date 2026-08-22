package ir

import (
	"fmt"
	"strings"
	"tach/foundation"
)

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

package ir

import (
	"fmt"
	"tach/foundation"
)

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

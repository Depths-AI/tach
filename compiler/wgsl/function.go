package wgsl

import (
	"fmt"
	"strings"
	"tach/foundation"
	"tach/ir"
)

type placeExpr struct {
	expr     string
	ty       *foundation.Type
	resource int
	index    string
}
type fnState struct {
	e       *emitter
	f       *ir.Function
	lowered *ir.Coordinates
	values  map[ir.ValueID]*foundation.Type
	places  map[ir.PlaceID]placeExpr
	loops   []*ir.Loop
}

func v(id ir.ValueID) string                             { return fmt.Sprintf("_v%d", id) }
func (s *fnState) def(id ir.ValueID, t *foundation.Type) { s.values[id] = t }
func (s *fnState) emitInstrs(b *ir.Block) error {
	for _, in := range b.Instrs {
		if err := s.emitInstr(in); err != nil {
			return err
		}
	}
	return nil
}

func (s *fnState) emitLoopTransfer(values []ir.ValueID, keyword string) error {
	if len(s.loops) == 0 {
		return fmt.Errorf("%s outside loop", keyword)
	}
	loop := s.loops[len(s.loops)-1]
	for i, id := range values {
		s.e.line("%s = %s;", v(loop.Results[i].ID), v(id))
	}
	s.e.line("%s;", keyword)
	return nil
}

func (s *fnState) emitBlock(b *ir.Block, yieldTargets []ir.Result) error {
	if err := s.emitInstrs(b); err != nil {
		return err
	}
	switch t := b.Term.(type) {
	case *ir.Return:
		if t.HasValue {
			s.e.line("return %s;", v(t.Value))
		} else {
			s.e.line("return;")
		}
	case *ir.Yield:
		for i, id := range t.Values {
			if i < len(yieldTargets) {
				s.e.line("%s = %s;", v(yieldTargets[i].ID), v(id))
			}
		}
	case *ir.Continue:
		return s.emitLoopTransfer(t.Values, "continue")
	case *ir.Break:
		return s.emitLoopTransfer(t.Values, "break")
	case *ir.Unreachable:
	case *ir.ExitScope:
		s.e.line("break;")
	default:
		return fmt.Errorf("unknown WGSL block terminator %T", b.Term)
	}
	return nil
}
func (s *fnState) emitInstr(in ir.Instr) error {
	e := s.e
	if definition, ok := in.(ir.ValueDef); ok && s.lowered != nil && s.lowered.Replaced[definition.ResultValue()] {
		return nil
	}
	switch x := in.(type) {
	case *ir.Const:
		if s.lowered != nil && s.lowered.Uses[x.Result] == 0 {
			return nil
		}
		raw := x.Raw
		if x.Type.Kind == foundation.Uint32Kind {
			raw += "u"
		}
		if x.Type.Kind == foundation.Int32Kind {
			raw += "i"
		}
		e.line("let %s: %s = %s;", v(x.Result), e.typeName(x.Type), raw)
		s.def(x.Result, x.Type)
	case *ir.Unary:
		e.line("let %s: %s = %s%s;", v(x.Result), e.typeName(x.Type), x.Op, v(x.X))
		s.def(x.Result, x.Type)
	case *ir.Binary:
		op := x.Op
		if op == "^" && foundation.IsBoolean(x.Type) {
			op = "!="
		}
		e.line("let %s: %s = %s %s %s;", v(x.Result), e.typeName(x.Type), v(x.Left), op, v(x.Right))
		s.def(x.Result, x.Type)
	case *ir.Intrinsic:
		args := make([]string, len(x.Args))
		for i, id := range x.Args {
			args[i] = v(id)
		}
		name := x.Kind.String()
		if x.Kind == ir.IntrinsicRSqrt {
			name = "inverseSqrt"
		}
		if x.Kind == ir.IntrinsicSelect {
			args = []string{args[2], args[1], args[0]}
		}
		switch x.Kind {
		case ir.IntrinsicMin:
			name, args = "select", []string{args[0], args[1], fmt.Sprintf("%s < %s", args[1], args[0])}
		case ir.IntrinsicMax:
			name, args = "select", []string{args[0], args[1], fmt.Sprintf("%s < %s", args[0], args[1])}
		case ir.IntrinsicClamp:
			bounded := v(x.Result) + "_bounded"
			e.line("let %s: %s = select(%s, %s, %s < %s);", bounded, e.typeName(x.Type), args[0], args[1], args[0], args[1])
			name, args = "select", []string{bounded, args[2], fmt.Sprintf("%s < %s", args[2], bounded)}
		}
		e.line("let %s: %s = %s(%s);", v(x.Result), e.typeName(x.Type), name, strings.Join(args, ", "))
		s.def(x.Result, x.Type)
	case *ir.Convert:
		e.line("let %s: %s = %s(%s);", v(x.Result), e.typeName(x.Type), e.typeName(x.Type), v(x.X))
		s.def(x.Result, x.Type)
	case *ir.Composite:
		args := make([]string, len(x.Values))
		for i, id := range x.Values {
			args[i] = v(id)
		}
		e.line("let %s: %s = %s(%s);", v(x.Result), e.typeName(x.Type), e.typeName(x.Type), strings.Join(args, ", "))
		s.def(x.Result, x.Type)
	case *ir.Extract:
		bt := s.values[x.Base]
		if bt == nil {
			return fmt.Errorf("WGSL extract base %%%d has no type", x.Base)
		}
		sel := ""
		if bt.Kind == foundation.StructKind {
			sel = fieldName(x.Index, bt.Fields[x.Index].Name)
		} else {
			sel = []string{"x", "y", "z", "w"}[x.Index]
		}
		e.line("let %s: %s = %s.%s;", v(x.Result), e.typeName(x.Type), v(x.Base), sel)
		s.def(x.Result, x.Type)
	case *ir.VectorIndex:
		e.line("let %s: %s = %s[%s];", v(x.Result), e.typeName(x.Type), v(x.Base), v(x.Index))
		s.def(x.Result, x.Type)
	case *ir.Call:
		args := make([]string, len(x.Args))
		for i, id := range x.Args {
			args[i] = v(id)
		}
		callee := s.e.m.Function(x.Function)
		if callee == nil {
			return fmt.Errorf("unknown callee %s", x.Function)
		}
		if x.Type.Kind == foundation.VoidKind {
			e.line("%s(%s);", s.e.funcName(callee), strings.Join(args, ", "))
		} else {
			e.line("let %s: %s = %s(%s);", v(x.Result), e.typeName(x.Type), s.e.funcName(callee), strings.Join(args, ", "))
			s.def(x.Result, x.Type)
		}
	case *ir.PlaceRoot:
		kernel := s.e.kernelIndex[s.f]
		name := resourceName(kernel, x.Buffer)
		if viewTexture(s.e.p.kernels[s.f], x.Buffer) {
			s.places[x.Result] = placeExpr{expr: name, ty: x.Type, resource: x.Buffer}
		} else if x.Type.Kind == foundation.StructKind && hasRuntimeTail(x.Type) {
			s.places[x.Result] = placeExpr{expr: name, ty: x.Type, resource: x.Buffer}
		} else {
			s.places[x.Result] = placeExpr{expr: name + ".data", ty: x.Type, resource: x.Buffer}
		}
	case *ir.PlaceWorkgroup:
		if x.Workgroup < 0 || x.Workgroup >= len(s.f.WorkgroupVars) {
			return fmt.Errorf("invalid workgroup place %d", x.Workgroup)
		}
		s.places[x.Result] = placeExpr{expr: workgroupName(s.f, x.Workgroup), ty: x.Type, resource: -1}
	case *ir.PlaceField:
		p := s.places[x.Base]
		if p.ty == nil {
			return fmt.Errorf("unknown place &p%d", x.Base)
		}
		name := fieldName(x.Field, p.ty.Fields[x.Field].Name)
		s.places[x.Result] = placeExpr{expr: p.expr + "." + name, ty: x.Type, resource: p.resource, index: p.index}
	case *ir.PlaceIndex:
		p := s.places[x.Base]
		if p.ty == nil {
			return fmt.Errorf("unknown place &p%d", x.Base)
		}
		s.places[x.Result] = placeExpr{expr: fmt.Sprintf("%s[%s]", p.expr, v(x.Index)), ty: x.Type, resource: p.resource, index: v(x.Index)}
	case *ir.Load:
		p := s.places[x.Place]
		if p.ty == nil {
			return fmt.Errorf("unknown load place &p%d", x.Place)
		}
		e.line("let %s: %s = %s;", v(x.Result), e.typeName(x.Type), p.expr)
		s.def(x.Result, x.Type)
	case *ir.Store:
		p := s.places[x.Place]
		if p.ty == nil {
			return fmt.Errorf("unknown store place &p%d", x.Place)
		}
		physical := s.e.p.kernels[s.f]
		if viewTexture(physical, p.resource) {
			if p.index == "" {
				return fmt.Errorf("view store has no pixel index")
			}
			xCoord, yCoord := p.index+" % "+v(physical.ViewWidth), p.index+" / "+v(physical.ViewWidth)
			if physical.Projection && len(s.f.Indices) >= 2 {
				xCoord, yCoord = v(s.f.Indices[0].ID), v(s.f.Indices[1].ID)
			}
			e.line("textureStore(%s, vec2<u32>(%s, %s), unpack4x8unorm(%s));", resourceName(s.e.kernelIndex[s.f], p.resource), xCoord, yCoord, v(x.Value))
		} else {
			e.line("%s = %s;", p.expr, v(x.Value))
		}
	case *ir.Atomic:
		p := s.places[x.Place]
		if p.ty == nil {
			return fmt.Errorf("unknown atomic place &p%d", x.Place)
		}
		name := ""
		switch x.Op {
		case ir.AtomicLoad:
			name = "atomicLoad"
		case ir.AtomicStore:
			name = "atomicStore"
		case ir.AtomicAdd:
			name = "atomicAdd"
		case ir.AtomicSub:
			name = "atomicSub"
		case ir.AtomicMin:
			name = "atomicMin"
		case ir.AtomicMax:
			name = "atomicMax"
		case ir.AtomicAnd:
			name = "atomicAnd"
		case ir.AtomicOr:
			name = "atomicOr"
		case ir.AtomicXor:
			name = "atomicXor"
		case ir.AtomicExchange:
			name = "atomicExchange"
		case ir.AtomicCompareExchange:
			name = "atomicCompareExchangeWeak"
		default:
			return fmt.Errorf("unknown atomic op %d", x.Op)
		}
		if x.Op == ir.AtomicCompareExchange {
			attempt := v(x.Result) + "_attempt"
			e.line("var %s: %s;", v(x.Result), e.typeName(x.Type))
			e.line("loop {")
			e.indent++
			e.line("let %s = %s(&%s, %s, %s);", attempt, name, p.expr, v(x.Expected), v(x.Value))
			e.line("%s = %s.old_value;", v(x.Result), attempt)
			e.line("if (%s.exchanged || %s.old_value != %s) { break; }", attempt, attempt, v(x.Expected))
			e.indent--
			e.line("}")
			s.def(x.Result, x.Type)
		} else if x.Op == ir.AtomicStore {
			e.line("%s(&%s, %s);", name, p.expr, v(x.Value))
		} else if x.Op == ir.AtomicLoad {
			e.line("let %s: %s = %s(&%s);", v(x.Result), e.typeName(x.Type), name, p.expr)
			s.def(x.Result, x.Type)
		} else {
			e.line("let %s: %s = %s(&%s, %s);", v(x.Result), e.typeName(x.Type), name, p.expr, v(x.Value))
			s.def(x.Result, x.Type)
		}
	case *ir.Barrier:
		switch x.Kind {
		case ir.BarrierWorkgroup:
			e.line("workgroupBarrier();")
		case ir.BarrierBuffer:
			e.line("storageBarrier();")
		default:
			return fmt.Errorf("unknown barrier kind %d", x.Kind)
		}
	case *ir.ArrayLength:
		p := s.places[x.Place]
		if p.ty == nil {
			return fmt.Errorf("unknown array length place &p%d", x.Place)
		}
		physical := s.e.p.kernels[s.f]
		if viewTexture(physical, p.resource) {
			e.line("let %s: u32 = %s * %s;", v(x.Result), v(physical.ViewWidth), v(physical.ViewHeight))
		} else if length, ok := physical.LogicalLengths[p.resource]; ok {
			e.line("let %s: u32 = %s;", v(x.Result), v(length))
		} else {
			e.line("let %s: u32 = arrayLength(&%s);", v(x.Result), p.expr)
		}
		s.def(x.Result, foundation.Uint32Type)
	case *ir.If:
		for _, r := range x.Results {
			e.line("var %s: %s;", v(r.ID), e.typeName(r.Type))
			s.def(r.ID, r.Type)
		}
		e.line("if (%s) {", v(x.Cond))
		e.indent++
		if err := s.emitBlock(x.Then, x.Results); err != nil {
			return err
		}
		e.indent--
		e.line("} else {")
		e.indent++
		if err := s.emitBlock(x.Else, x.Results); err != nil {
			return err
		}
		e.indent--
		e.line("}")
	case *ir.Loop:
		for i, r := range x.Results {
			e.line("var %s: %s = %s;", v(r.ID), e.typeName(r.Type), v(x.Params[i].Init))
			s.def(r.ID, r.Type)
			s.def(x.Params[i].ID, x.Params[i].Type)
		}
		e.line("loop {")
		e.indent++
		s.loops = append(s.loops, x)
		for i, p := range x.Params {
			e.line("let %s: %s = %s;", v(p.ID), e.typeName(p.Type), v(x.Results[i].ID))
		}
		if err := s.emitInstrs(x.Cond); err != nil {
			return err
		}
		cy, ok := x.Cond.Term.(*ir.Yield)
		if !ok || len(cy.Values) != 1 {
			return fmt.Errorf("loop condition malformed")
		}
		e.line("if (!%s) { break; }", v(cy.Values[0]))
		err := s.emitBlock(x.Body, nil)
		s.loops = s.loops[:len(s.loops)-1]
		if err != nil {
			return err
		}
		e.indent--
		e.line("}")
	case *ir.Scope:
		e.line("loop {")
		e.indent++
		if err := s.emitBlock(x.Body, nil); err != nil {
			return err
		}
		if _, exits := x.Body.Term.(*ir.ExitScope); !exits {
			if _, returns := x.Body.Term.(*ir.Return); !returns {
				e.line("break;")
			}
		}
		e.indent--
		e.line("}")
	default:
		return fmt.Errorf("unknown WGSL instruction %T", in)
	}
	return nil
}

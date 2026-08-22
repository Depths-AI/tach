package ir

import (
	"fmt"
	"maps"

	"tach/src/foundation"
)

type placeInfo struct {
	ty     *foundation.Type
	buffer int
}

type verifyEnv struct {
	values map[ValueID]*foundation.Type
	places map[PlaceID]placeInfo
}

func (e verifyEnv) clone() verifyEnv {
	return verifyEnv{maps.Clone(e.values), maps.Clone(e.places)}
}

func Verify(m *Module) error {
	if m == nil {
		return fmt.Errorf("nil IR module")
	}
	names := map[string]bool{}
	for _, s := range m.Structs {
		if s.Kind != foundation.StructKind {
			return fmt.Errorf("module struct %s is not a struct", s)
		}
		if names[s.Name] {
			return fmt.Errorf("duplicate type %q", s.Name)
		}
		names[s.Name] = true
		fn := map[string]bool{}
		for _, f := range s.Fields {
			if fn[f.Name] {
				return fmt.Errorf("duplicate field %s.%s", s.Name, f.Name)
			}
			fn[f.Name] = true
		}
	}
	fmap := map[string]*Function{}
	for _, f := range m.Functions {
		if _, ok := fmap[f.Name]; ok {
			return fmt.Errorf("duplicate function %q", f.Name)
		}
		fmap[f.Name] = f
	}
	for _, f := range m.Functions {
		if err := verifyFunction(m, f, fmap); err != nil {
			return fmt.Errorf("%s: %w", f.Name, err)
		}
	}
	return nil
}

func verifyUniqueIDs(f *Function) error {
	values := map[ValueID]string{}
	places := map[PlaceID]string{}
	defValue := func(id ValueID, where string) error {
		if id == 0 {
			return fmt.Errorf("value id 0 is reserved (%s)", where)
		}
		if prev, ok := values[id]; ok {
			return fmt.Errorf("value %%%d is defined twice: %s and %s", id, prev, where)
		}
		values[id] = where
		return nil
	}
	defPlace := func(id PlaceID, where string) error {
		if id == 0 {
			return fmt.Errorf("place id 0 is reserved (%s)", where)
		}
		if prev, ok := places[id]; ok {
			return fmt.Errorf("place &p%d is defined twice: %s and %s", id, prev, where)
		}
		places[id] = where
		return nil
	}
	for _, p := range f.Params {
		if err := defValue(p.ID, "parameter "+p.Name); err != nil {
			return err
		}
	}
	for _, p := range f.Indices {
		if err := defValue(p.ID, "logical index "+p.Name); err != nil {
			return err
		}
	}
	var walk func(*Block, string) error
	walk = func(b *Block, region string) error {
		if b == nil {
			return fmt.Errorf("nil block in %s", region)
		}
		for i, in := range b.Instrs {
			where := fmt.Sprintf("%s instruction %d (%T)", region, i, in)
			if vd, ok := in.(ValueDef); ok && vd.ResultValue() != 0 {
				if err := defValue(vd.ResultValue(), where); err != nil {
					return err
				}
			}
			if pd, ok := in.(PlaceDef); ok {
				if err := defPlace(pd.ResultPlace(), where); err != nil {
					return err
				}
			}
			switch x := in.(type) {
			case *If:
				for j, r := range x.Results {
					if err := defValue(r.ID, fmt.Sprintf("%s if-result %d", region, j)); err != nil {
						return err
					}
				}
				if err := walk(x.Then, region+"/if.then"); err != nil {
					return err
				}
				if err := walk(x.Else, region+"/if.else"); err != nil {
					return err
				}
			case *Loop:
				for j, p := range x.Params {
					if err := defValue(p.ID, fmt.Sprintf("%s loop-param %d", region, j)); err != nil {
						return err
					}
				}
				for j, r := range x.Results {
					if err := defValue(r.ID, fmt.Sprintf("%s loop-result %d", region, j)); err != nil {
						return err
					}
				}
				if err := walk(x.Cond, region+"/loop.cond"); err != nil {
					return err
				}
				if err := walk(x.Body, region+"/loop.body"); err != nil {
					return err
				}
			case *Scope:
				if err := walk(x.Body, region+"/scope"); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return walk(f.Body, "function "+f.Name)
}

func verifyFunction(m *Module, f *Function, fmap map[string]*Function) error {
	if err := verifyUniqueIDs(f); err != nil {
		return err
	}
	if f.Return == nil {
		return fmt.Errorf("missing return type")
	}
	if f.Kind == Stage {
		if len(f.Indices) < 1 || len(f.Indices) > 3 {
			return fmt.Errorf("compute function requires 1 to 3 logical indices")
		}
		for _, index := range f.Indices {
			if !foundation.Equal(index.Type, foundation.Uint32Type) {
				return fmt.Errorf("logical index %s has type %s, want uint32", index.Name, index.Type)
			}
		}
		if f.Return.Kind != foundation.VoidKind {
			return fmt.Errorf("compute function must return void")
		}
		if f.Workgroup.Explicit {
			for _, n := range f.Workgroup.Size {
				if n == 0 {
					return fmt.Errorf("workgroup size must be positive")
				}
			}
		}
	} else {
		if f.Kind != Helper {
			return fmt.Errorf("invalid function kind")
		}
		if len(f.Indices) != 0 {
			return fmt.Errorf("helper function cannot have logical indices")
		}
		if len(f.WorkgroupVars) != 0 {
			return fmt.Errorf("helper function cannot declare workgroup variables")
		}
		if len(f.SourceParams) != 0 || len(f.BufferParams) != 0 {
			return fmt.Errorf("helper function cannot have stage parameters")
		}
	}
	valueParams := map[ValueID]Param{}
	for _, p := range f.Params {
		valueParams[p.ID] = p
		if f.Kind == Stage && !foundation.IsConstructible(p.Type) {
			return fmt.Errorf("kernel parameter %s has invalid value type %s", p.Name, p.Type)
		}
	}
	if f.Kind == Stage {
		seenNames := map[string]bool{}
		seenValues := map[ValueID]bool{}
		seenBuffers := map[int]bool{}
		for _, p := range f.SourceParams {
			if p.Name == "" || seenNames[p.Name] {
				return fmt.Errorf("invalid or duplicate kernel parameter name %q", p.Name)
			}
			seenNames[p.Name] = true
			if p.Kind == SourceValue {
				value, ok := valueParams[p.Value]
				if !ok || p.Buffer != -1 || seenValues[p.Value] || value.Name != p.Name {
					return fmt.Errorf("kernel value parameter %s has invalid mapping", p.Name)
				}
				seenValues[p.Value] = true
			} else if p.Kind == SourceBuffer {
				if p.Buffer < 0 || p.Buffer >= len(f.BufferParams) || seenBuffers[p.Buffer] || f.BufferParams[p.Buffer].Name != p.Name {
					return fmt.Errorf("stage buffer parameter %s has invalid mapping", p.Name)
				}
				seenBuffers[p.Buffer] = true
			} else {
				return fmt.Errorf("stage parameter %s has invalid kind", p.Name)
			}
		}
		if len(seenBuffers) == 0 {
			return fmt.Errorf("kernel requires at least one buffer parameter")
		}
		for _, parameter := range f.Params {
			if !seenValues[parameter.ID] {
				return fmt.Errorf("kernel value parameter %s is not mapped", parameter.Name)
			}
		}
		for i, buffer := range f.BufferParams {
			if buffer.Name == "" || !foundation.IsHostShareable(buffer.Type) || (buffer.Access != Read && buffer.Access != Mutable) || !seenBuffers[i] {
				return fmt.Errorf("invalid stage buffer parameter %d", i)
			}
		}
	}
	wgnames := map[string]bool{}
	for i, w := range f.WorkgroupVars {
		if f.Kind != Stage {
			return fmt.Errorf("workgroup variable %d used outside compute function", i)
		}
		if w.Name == "" || wgnames[w.Name] {
			return fmt.Errorf("invalid or duplicate workgroup variable name %q", w.Name)
		}
		wgnames[w.Name] = true
		if !foundation.IsWorkgroupStorable(w.Type) {
			return fmt.Errorf("workgroup variable %s has invalid type %s", w.Name, w.Type)
		}
	}
	e := verifyEnv{values: map[ValueID]*foundation.Type{}, places: map[PlaceID]placeInfo{}}
	for _, index := range f.Indices {
		e.values[index.ID] = index.Type
	}
	for _, p := range f.Params {
		if p.ID == 0 {
			return fmt.Errorf("value id 0 is reserved")
		}
		if _, ok := e.values[p.ID]; ok {
			return fmt.Errorf("duplicate parameter value %%%d", p.ID)
		}
		e.values[p.ID] = p.Type
	}
	_, err := verifyBlock(m, f, f.Body, e, fmap, "return", nil)
	if err != nil {
		return err
	}
	if f.Kind == Stage {
		if err := verifyUniformity(m, f, fmap); err != nil {
			return err
		}
	}
	return nil
}

func verifyBlock(m *Module, f *Function, b *Block, e verifyEnv, fmap map[string]*Function, termKind string, loopTypes []*foundation.Type) (verifyEnv, error) {
	if b == nil {
		return e, fmt.Errorf("nil block")
	}
	defVal := func(id ValueID, t *foundation.Type) error {
		if id == 0 {
			return fmt.Errorf("value id 0 is reserved")
		}
		if _, ok := e.values[id]; ok {
			return fmt.Errorf("value %%%d redefined", id)
		}
		e.values[id] = t
		return nil
	}
	defPlace := func(id PlaceID, p placeInfo) error {
		if id == 0 {
			return fmt.Errorf("place id 0 is reserved")
		}
		if _, ok := e.places[id]; ok {
			return fmt.Errorf("place &p%d redefined", id)
		}
		e.places[id] = p
		return nil
	}
	val := func(id ValueID) (*foundation.Type, error) {
		t, ok := e.values[id]
		if !ok {
			return nil, fmt.Errorf("use of undefined value %%%d", id)
		}
		return t, nil
	}
	place := func(id PlaceID) (placeInfo, error) {
		p, ok := e.places[id]
		if !ok {
			return placeInfo{}, fmt.Errorf("use of undefined place &p%d", id)
		}
		return p, nil
	}
	for _, in := range b.Instrs {
		switch x := in.(type) {
		case *Const:
			if !foundation.IsScalar(x.Type) {
				return e, fmt.Errorf("constant %%%d has non-scalar type %s", x.Result, x.Type)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Unary:
			t, err := val(x.X)
			if err != nil {
				return e, err
			}
			if !foundation.Equal(t, x.Type) {
				return e, fmt.Errorf("unary %s type mismatch %s -> %s", x.Op, t, x.Type)
			}
			if x.Op == "!" && !foundation.IsBoolean(t) {
				return e, fmt.Errorf("! requires bool or boolean vector")
			}
			if x.Op == "-" && !foundation.IsSignedNumeric(t) {
				return e, fmt.Errorf("unary - requires signed/float numeric")
			}
			if x.Op == "~" && !foundation.IsIntegerLike(t) {
				return e, fmt.Errorf("unary ~ requires integer scalar/vector")
			}
			if x.Op != "!" && x.Op != "-" && x.Op != "~" {
				return e, fmt.Errorf("unknown unary op %q", x.Op)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Binary:
			lt, err := val(x.Left)
			if err != nil {
				return e, err
			}
			rt, err := val(x.Right)
			if err != nil {
				return e, err
			}
			if err := verifyBinary(x, lt, rt); err != nil {
				return e, err
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Convert:
			t, err := val(x.X)
			if err != nil {
				return e, err
			}
			if !foundation.Equal(t, x.From) {
				return e, fmt.Errorf("convert source says %s but value is %s", x.From, t)
			}
			if !foundation.IsNumericScalar(t) || !foundation.IsNumericScalar(x.Type) {
				return e, fmt.Errorf("convert supports scalar numerics")
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Composite:
			if x.Type.Kind == foundation.StructKind {
				if len(x.Values) != len(x.Type.Fields) {
					return e, fmt.Errorf("struct %s construction has %d values", x.Type, len(x.Values))
				}
				for i, id := range x.Values {
					t, err := val(id)
					if err != nil {
						return e, err
					}
					if !foundation.Equal(t, x.Type.Fields[i].Type) {
						return e, fmt.Errorf("struct %s field %s has %s, want %s", x.Type, x.Type.Fields[i].Name, t, x.Type.Fields[i].Type)
					}
				}
			} else if x.Type.Kind == foundation.VectorKind {
				if len(x.Values) != x.Type.Lanes {
					return e, fmt.Errorf("vector construction has %d components, want %d", len(x.Values), x.Type.Lanes)
				}
				for _, id := range x.Values {
					t, err := val(id)
					if err != nil {
						return e, err
					}
					if !foundation.Equal(t, x.Type.Elem) {
						return e, fmt.Errorf("vector component has %s, want %s", t, x.Type.Elem)
					}
				}
			} else {
				return e, fmt.Errorf("composite result type %s is not constructible", x.Type)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Extract:
			bt, err := val(x.Base)
			if err != nil {
				return e, err
			}
			var et *foundation.Type
			if bt.Kind == foundation.StructKind {
				if x.Index < 0 || x.Index >= len(bt.Fields) {
					return e, fmt.Errorf("struct extract index %d out of range", x.Index)
				}
				et = bt.Fields[x.Index].Type
			} else if bt.Kind == foundation.VectorKind {
				if x.Index < 0 || x.Index >= bt.Lanes {
					return e, fmt.Errorf("vector extract index %d out of range", x.Index)
				}
				et = bt.Elem
			} else {
				return e, fmt.Errorf("extract from %s", bt)
			}
			if !foundation.Equal(et, x.Type) {
				return e, fmt.Errorf("extract type %s, want %s", x.Type, et)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *VectorIndex:
			bt, err := val(x.Base)
			if err != nil {
				return e, err
			}
			it, err := val(x.Index)
			if err != nil {
				return e, err
			}
			if bt.Kind != foundation.VectorKind {
				return e, fmt.Errorf("vector index base is %s", bt)
			}
			if !foundation.IsInteger(it) {
				return e, fmt.Errorf("vector index is %s, want int32 or uint32", it)
			}
			if !foundation.Equal(x.Type, bt.Elem) {
				return e, fmt.Errorf("vector index type %s, want %s", x.Type, bt.Elem)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Intrinsic:
			args := make([]*foundation.Type, len(x.Args))
			for i, id := range x.Args {
				t, err := val(id)
				if err != nil {
					return e, err
				}
				args[i] = t
			}
			if err := verifyIntrinsic(x, args); err != nil {
				return e, err
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Call:
			callee := fmap[x.Function]
			if callee == nil {
				return e, fmt.Errorf("call to unknown function %s", x.Function)
			}
			if callee.Kind == Stage {
				return e, fmt.Errorf("compute entry point %s cannot be called", x.Function)
			}
			if len(x.Args) != len(callee.Params) {
				return e, fmt.Errorf("call %s has %d args, want %d", x.Function, len(x.Args), len(callee.Params))
			}
			for i, id := range x.Args {
				t, err := val(id)
				if err != nil {
					return e, err
				}
				if !foundation.Equal(t, callee.Params[i].Type) {
					return e, fmt.Errorf("call %s arg %d is %s, want %s", x.Function, i, t, callee.Params[i].Type)
				}
			}
			if !foundation.Equal(x.Type, callee.Return) {
				return e, fmt.Errorf("call %s result says %s, want %s", x.Function, x.Type, callee.Return)
			}
			if x.Type.Kind != foundation.VoidKind {
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			}
		case *PlaceRoot:
			if x.Buffer < 0 || x.Buffer >= len(f.BufferParams) {
				return e, fmt.Errorf("invalid buffer %d", x.Buffer)
			}
			r := f.BufferParams[x.Buffer]
			if !foundation.Equal(x.Type, r.Type) {
				return e, fmt.Errorf("buffer place type %s, want %s", x.Type, r.Type)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, x.Buffer}); err != nil {
				return e, err
			}
		case *PlaceWorkgroup:
			if f.Kind != Stage || x.Workgroup < 0 || x.Workgroup >= len(f.WorkgroupVars) {
				return e, fmt.Errorf("invalid workgroup place %d", x.Workgroup)
			}
			w := f.WorkgroupVars[x.Workgroup]
			if !foundation.Equal(x.Type, w.Type) {
				return e, fmt.Errorf("workgroup place type %s, want %s", x.Type, w.Type)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, -1}); err != nil {
				return e, err
			}
		case *PlaceField:
			bp, err := place(x.Base)
			if err != nil {
				return e, err
			}
			if bp.ty.Kind != foundation.StructKind || x.Field < 0 || x.Field >= len(bp.ty.Fields) {
				return e, fmt.Errorf("invalid field place on %s", bp.ty)
			}
			want := bp.ty.Fields[x.Field].Type
			if !foundation.Equal(want, x.Type) {
				return e, fmt.Errorf("field place type %s, want %s", x.Type, want)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, bp.buffer}); err != nil {
				return e, err
			}
		case *PlaceIndex:
			bp, err := place(x.Base)
			if err != nil {
				return e, err
			}
			if bp.ty.Kind != foundation.RuntimeArrayKind && bp.ty.Kind != foundation.FixedArrayKind && bp.ty.Kind != foundation.VectorKind {
				return e, fmt.Errorf("index place base is %s", bp.ty)
			}
			it, err := val(x.Index)
			if err != nil {
				return e, err
			}
			if !foundation.Equal(it, foundation.Uint32Type) && !foundation.Equal(it, foundation.Int32Type) {
				return e, fmt.Errorf("array index is %s", it)
			}
			if !foundation.Equal(x.Type, bp.ty.Elem) {
				return e, fmt.Errorf("index result %s, want %s", x.Type, bp.ty.Elem)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, bp.buffer}); err != nil {
				return e, err
			}
		case *Load:
			p, err := place(x.Place)
			if err != nil {
				return e, err
			}
			if !foundation.IsConstructible(p.ty) {
				return e, fmt.Errorf("place of type %s cannot be loaded as a value", p.ty)
			}
			if !foundation.Equal(p.ty, x.Type) {
				return e, fmt.Errorf("load type %s, place is %s", x.Type, p.ty)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Store:
			p, err := place(x.Place)
			if err != nil {
				return e, err
			}
			v, err := val(x.Value)
			if err != nil {
				return e, err
			}
			if !foundation.Equal(p.ty, v) {
				return e, fmt.Errorf("store %s into %s", v, p.ty)
			}
			if !foundation.IsConstructible(p.ty) {
				return e, fmt.Errorf("place of type %s cannot be stored as a whole value", p.ty)
			}
			if p.buffer >= 0 {
				r := f.BufferParams[p.buffer]
				if r.Access != Mutable {
					return e, fmt.Errorf("store through non-writable buffer %s", r.Name)
				}
			}
		case *Atomic:
			p, err := place(x.Place)
			if err != nil {
				return e, err
			}
			if p.ty.Kind != foundation.AtomicKind || !foundation.Equal(p.ty.Elem, x.Type) || (x.Type.Kind != foundation.Int32Kind && x.Type.Kind != foundation.Uint32Kind) {
				return e, fmt.Errorf("atomic operation type %s does not match place %s", x.Type, p.ty)
			}
			if p.buffer >= 0 && x.Op != AtomicLoad {
				r := f.BufferParams[p.buffer]
				if r.Access != Mutable {
					return e, fmt.Errorf("atomic operation through non-writable buffer resource %s", r.Name)
				}
			}
			switch x.Op {
			case AtomicLoad:
				if x.Result == 0 || x.Value != 0 || x.Expected != 0 {
					return e, fmt.Errorf("atomicLoad result/value shape is invalid")
				}
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			case AtomicStore:
				if x.Result != 0 || x.Value == 0 || x.Expected != 0 {
					return e, fmt.Errorf("atomicStore result/value shape is invalid")
				}
				vt, err := val(x.Value)
				if err != nil || !foundation.Equal(vt, x.Type) {
					if err != nil {
						return e, err
					}
					return e, fmt.Errorf("atomicStore value is %s, want %s", vt, x.Type)
				}
			case AtomicAdd, AtomicSub, AtomicMin, AtomicMax, AtomicAnd, AtomicOr, AtomicXor, AtomicExchange:
				if x.Result == 0 || x.Value == 0 || x.Expected != 0 {
					return e, fmt.Errorf("atomic read-modify-write result/value shape is invalid")
				}
				vt, err := val(x.Value)
				if err != nil {
					return e, err
				}
				if !foundation.Equal(vt, x.Type) {
					return e, fmt.Errorf("atomic operand is %s, want %s", vt, x.Type)
				}
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			case AtomicCompareExchange:
				if x.Result == 0 || x.Value == 0 || x.Expected == 0 {
					return e, fmt.Errorf("atomicCompareExchange result/operand shape is invalid")
				}
				for _, operand := range []ValueID{x.Expected, x.Value} {
					operandType, err := val(operand)
					if err != nil {
						return e, err
					}
					if !foundation.Equal(operandType, x.Type) {
						return e, fmt.Errorf("atomic compare-exchange operand is %s, want %s", operandType, x.Type)
					}
				}
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			default:
				return e, fmt.Errorf("unknown atomic operation %d", x.Op)
			}
		case *Barrier:
			if f.Kind != Stage {
				return e, fmt.Errorf("barrier outside compute function")
			}
			if x.Kind != BarrierWorkgroup && x.Kind != BarrierBuffer {
				return e, fmt.Errorf("unknown barrier kind %d", x.Kind)
			}
		case *ArrayLength:
			p, err := place(x.Place)
			if err != nil {
				return e, err
			}
			if p.ty.Kind != foundation.RuntimeArrayKind {
				return e, fmt.Errorf("array_length on %s", p.ty)
			}
			if !foundation.Equal(x.Type, foundation.Uint32Type) {
				return e, fmt.Errorf("array_length result must be uint32")
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *If:
			ct, err := val(x.Cond)
			if err != nil {
				return e, err
			}
			if !foundation.Equal(ct, foundation.BoolType) {
				return e, fmt.Errorf("if condition is %s", ct)
			}
			te, err := verifyBlock(m, f, x.Then, e.clone(), fmap, "yield", loopTypes)
			if err != nil {
				return e, fmt.Errorf("if then: %w", err)
			}
			ee, err := verifyBlock(m, f, x.Else, e.clone(), fmap, "yield", loopTypes)
			if err != nil {
				return e, fmt.Errorf("if else: %w", err)
			}
			ty, ok1 := x.Then.Term.(*Yield)
			ey, ok2 := x.Else.Term.(*Yield)
			if len(x.Results) > 0 && !ok1 && !ok2 {
				return e, fmt.Errorf("value-producing if has no continuing branch")
			}
			if ok1 && len(ty.Values) != len(x.Results) {
				return e, fmt.Errorf("if then yield arity mismatch")
			}
			if ok2 && len(ey.Values) != len(x.Results) {
				return e, fmt.Errorf("if else yield arity mismatch")
			}
			for i, r := range x.Results {
				if ok1 {
					a := te.values[ty.Values[i]]
					if !foundation.Equal(a, r.Type) {
						return e, fmt.Errorf("if then result %d type mismatch", i)
					}
				}
				if ok2 {
					bb := ee.values[ey.Values[i]]
					if !foundation.Equal(bb, r.Type) {
						return e, fmt.Errorf("if else result %d type mismatch", i)
					}
				}
				if err := defVal(r.ID, r.Type); err != nil {
					return e, err
				}
			}
		case *Loop:
			le := e.clone()
			for _, p := range x.Params {
				it, err := val(p.Init)
				if err != nil {
					return e, err
				}
				if !foundation.Equal(it, p.Type) {
					return e, fmt.Errorf("loop init %s, want %s", it, p.Type)
				}
				if _, exists := le.values[p.ID]; exists {
					return e, fmt.Errorf("loop param %%%d redefined", p.ID)
				}
				le.values[p.ID] = p.Type
			}
			ce, err := verifyBlock(m, f, x.Cond, le.clone(), fmap, "yield", nil)
			if err != nil {
				return e, fmt.Errorf("loop condition: %w", err)
			}
			cy, ok := x.Cond.Term.(*Yield)
			if !ok || len(cy.Values) != 1 || !foundation.Equal(ce.values[cy.Values[0]], foundation.BoolType) {
				return e, fmt.Errorf("loop condition must yield one bool")
			}
			carriedTypes := make([]*foundation.Type, len(x.Params))
			for i, parameter := range x.Params {
				carriedTypes[i] = parameter.Type
			}
			_, err = verifyBlock(m, f, x.Body, le.clone(), fmap, "continue", carriedTypes)
			if err != nil {
				return e, fmt.Errorf("loop body: %w", err)
			}
			if len(x.Results) != len(x.Params) {
				return e, fmt.Errorf("loop carried arity mismatch")
			}
			for i, p := range x.Params {
				if !foundation.Equal(x.Results[i].Type, p.Type) {
					return e, fmt.Errorf("loop carried value %d type mismatch", i)
				}
				if err := defVal(x.Results[i].ID, p.Type); err != nil {
					return e, err
				}
			}
		case *Scope:
			if f.Kind != Stage {
				return e, fmt.Errorf("scope outside stage")
			}
			if _, err := verifyBlock(m, f, x.Body, e.clone(), fmap, "exit_scope", loopTypes); err != nil {
				return e, fmt.Errorf("scope: %w", err)
			}
		default:
			return e, fmt.Errorf("unknown instruction %T", in)
		}
	}
	if b.Term == nil {
		return e, fmt.Errorf("block has no terminator")
	}
	verifyTransfer := func(kind string, values []ValueID) error {
		if loopTypes == nil {
			return fmt.Errorf("unexpected %s terminator", kind)
		}
		if len(values) != len(loopTypes) {
			return fmt.Errorf("%s carries %d values, want %d", kind, len(values), len(loopTypes))
		}
		for i, id := range values {
			type_, err := val(id)
			if err != nil {
				return err
			}
			if !foundation.Equal(type_, loopTypes[i]) {
				return fmt.Errorf("%s value %d is %s, want %s", kind, i, type_, loopTypes[i])
			}
		}
		return nil
	}
	switch t := b.Term.(type) {
	case *Return:
		if f.Return.Kind == foundation.VoidKind {
			if t.HasValue {
				return e, fmt.Errorf("void function returns a value")
			}
		} else {
			if !t.HasValue {
				return e, fmt.Errorf("function returning %s has bare return", f.Return)
			}
			vt, err := val(t.Value)
			if err != nil {
				return e, err
			}
			if !foundation.Equal(vt, f.Return) {
				return e, fmt.Errorf("return value is %s, want %s", vt, f.Return)
			}
		}
	case *Yield:
		if termKind != "yield" {
			return e, fmt.Errorf("unexpected yield terminator")
		}
		for _, id := range t.Values {
			if _, err := val(id); err != nil {
				return e, err
			}
		}
	case *Continue:
		if err := verifyTransfer("continue", t.Values); err != nil {
			return e, err
		}
	case *Break:
		if err := verifyTransfer("break", t.Values); err != nil {
			return e, err
		}
	case *Unreachable:
		// Structured constructs whose every path exits can leave an unreachable merge.
	case *ExitScope:
		if termKind != "exit_scope" && termKind != "yield" && termKind != "continue" {
			return e, fmt.Errorf("exit_scope outside scope")
		}
	default:
		return e, fmt.Errorf("unknown terminator %T", b.Term)
	}
	return e, nil
}

func verifyIntrinsic(x *Intrinsic, args []*foundation.Type) error {
	if x.Kind == IntrinsicAll || x.Kind == IntrinsicAny {
		if len(args) != 1 || args[0] == nil || args[0].Kind != foundation.VectorKind || args[0].Elem.Kind != foundation.BoolKind || !foundation.Equal(x.Type, foundation.BoolType) {
			return fmt.Errorf("intrinsic %s requires vec<bool, N> and returns bool", x.Kind)
		}
		return nil
	}
	if x.Kind == IntrinsicSelect {
		if len(args) != 3 || args[0] == nil || args[0].Kind != foundation.VectorKind || args[0].Elem.Kind != foundation.BoolKind || !foundation.Equal(args[1], args[2]) || !foundation.Equal(x.Type, args[1]) || x.Type.Kind != foundation.VectorKind || x.Type.Lanes != args[0].Lanes {
			return fmt.Errorf("intrinsic select requires a boolean-vector mask and matching vector arms")
		}
		return nil
	}
	rule := x.Kind.Rule()
	if rule.Arity == 0 {
		return fmt.Errorf("unknown intrinsic %d", x.Kind)
	}
	if len(args) != rule.Arity {
		return fmt.Errorf("intrinsic %s has %d args, want %d", x.Kind, len(args), rule.Arity)
	}
	t := args[0]
	element := t
	lanes := 0
	if t != nil && t.Kind == foundation.VectorKind {
		element, lanes = t.Elem, t.Lanes
	}
	if !rule.Domain.Accepts(element) || rule.VectorOnly && lanes == 0 || rule.Lanes != 0 && lanes != rule.Lanes {
		return fmt.Errorf("intrinsic %s does not accept %s", x.Kind, t)
	}
	for _, argument := range args[1:] {
		if !foundation.Equal(argument, t) {
			return fmt.Errorf("intrinsic %s requires matching operands", x.Kind)
		}
	}
	out := t
	if rule.ResultElement {
		out = element
	}
	if !foundation.Equal(x.Type, out) {
		return fmt.Errorf("intrinsic %s returns %s, got %s", x.Kind, out, x.Type)
	}
	return nil
}

func verifyBinary(x *Binary, l, r *foundation.Type) error {
	switch x.Op {
	case "+", "-":
		if !foundation.Equal(l, r) || !foundation.Equal(x.Type, l) || !foundation.IsNumeric(l) {
			return fmt.Errorf("%s requires matching numeric operands; got %s and %s -> %s", x.Op, l, r, x.Type)
		}
	case "*":
		if foundation.Equal(l, r) && foundation.Equal(x.Type, l) && foundation.IsNumeric(l) {
			return nil
		}
		if l.Kind == foundation.VectorKind && foundation.Equal(r, l.Elem) && foundation.Equal(x.Type, l) {
			return nil
		}
		if r.Kind == foundation.VectorKind && foundation.Equal(l, r.Elem) && foundation.Equal(x.Type, r) {
			return nil
		}
		return fmt.Errorf("* invalid for %s and %s -> %s", l, r, x.Type)
	case "/", "%":
		if foundation.Equal(l, r) && foundation.Equal(x.Type, l) && foundation.IsNumeric(l) && (x.Op == "/" || foundation.IsNumericScalar(l)) {
			return nil
		}
		if x.Op == "/" && l.Kind == foundation.VectorKind && foundation.Equal(r, l.Elem) && foundation.Equal(x.Type, l) {
			return nil
		}
		return fmt.Errorf("%s invalid for %s and %s", x.Op, l, r)
	case "==", "!=", "<", "<=", ">", ">=":
		valid := foundation.IsNumeric(l) || (x.Op == "==" || x.Op == "!=") && foundation.IsBoolean(l)
		if !foundation.Equal(l, r) || !valid || !foundation.Equal(x.Type, foundation.BoolShape(l)) {
			return fmt.Errorf("comparison invalid for %s and %s", l, r)
		}
	case "&&", "||":
		if !foundation.Equal(l, foundation.BoolType) || !foundation.Equal(r, foundation.BoolType) || !foundation.Equal(x.Type, foundation.BoolType) {
			return fmt.Errorf("logical op requires bool operands")
		}
	case "&", "|", "^":
		if !foundation.Equal(l, r) || !foundation.Equal(x.Type, l) || !foundation.IsIntegerLike(l) && !foundation.IsBoolean(l) {
			return fmt.Errorf("%s requires matching integer or boolean operands; got %s and %s -> %s", x.Op, l, r, x.Type)
		}
	case "<<", ">>":
		want := foundation.ShiftCountType(l)
		if want == nil || !foundation.Equal(r, want) || !foundation.Equal(x.Type, l) {
			return fmt.Errorf("%s requires integer value %s shifted by %s; got %s and %s -> %s", x.Op, l, want, l, r, x.Type)
		}
	default:
		return fmt.Errorf("unknown binary op %q", x.Op)
	}
	return nil
}

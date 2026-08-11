package ir

import (
	"fmt"

	"tach/src/types"
)

type placeInfo struct {
	ty       *types.Type
	resource int
}

type verifyEnv struct {
	values map[ValueID]*types.Type
	places map[PlaceID]placeInfo
}

func (e verifyEnv) clone() verifyEnv {
	v := make(map[ValueID]*types.Type, len(e.values))
	for k, x := range e.values {
		v[k] = x
	}
	p := make(map[PlaceID]placeInfo, len(e.places))
	for k, x := range e.places {
		p[k] = x
	}
	return verifyEnv{v, p}
}

func Verify(m *Module) error {
	if m == nil {
		return fmt.Errorf("nil IR module")
	}
	names := map[string]bool{}
	for _, s := range m.Structs {
		if s.Kind != types.Struct {
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
	for i, r := range m.Resources {
		if r.Name == "" {
			return fmt.Errorf("resource %d has empty name", i)
		}
		if r.Kind != Uniform && r.Kind != Buffer {
			return fmt.Errorf("resource %s has invalid kind", r.Name)
		}
		if r.Kind == Uniform && types.ContainsRuntimeArray(r.Type) {
			return fmt.Errorf("uniform resource %s cannot contain a runtime-sized array", r.Name)
		}
		if r.Kind == Uniform && types.ContainsAtomic(r.Type) {
			return fmt.Errorf("uniform resource %s cannot contain atomic values", r.Name)
		}
		if !types.IsHostShareable(r.Type) {
			return fmt.Errorf("resource %s type %s is not host-shareable", r.Name, r.Type)
		}
		if r.Kind == Uniform && r.Access != Read {
			return fmt.Errorf("uniform resource %s must be read-only", r.Name)
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
	if f.Compute {
		if len(f.Params) != 0 {
			return fmt.Errorf("compute function cannot have value parameters")
		}
		if len(f.Indices) < 1 || len(f.Indices) > 3 {
			return fmt.Errorf("compute function requires 1 to 3 logical indices")
		}
		for _, index := range f.Indices {
			if !types.Equal(index.Type, types.TU32) {
				return fmt.Errorf("logical index %s has type %s, want u32", index.Name, index.Type)
			}
		}
		if f.Return.Kind != types.Void {
			return fmt.Errorf("compute function must return void")
		}
		for _, n := range f.Workgroup {
			if n == 0 {
				return fmt.Errorf("workgroup size must be positive")
			}
		}
	} else {
		if len(f.Indices) != 0 {
			return fmt.Errorf("helper function cannot have logical indices")
		}
		if len(f.WorkgroupVars) != 0 {
			return fmt.Errorf("helper function cannot declare workgroup variables")
		}
	}
	wgnames := map[string]bool{}
	for i, w := range f.WorkgroupVars {
		if !f.Compute {
			return fmt.Errorf("workgroup variable %d used outside compute function", i)
		}
		if w.Name == "" || wgnames[w.Name] {
			return fmt.Errorf("invalid or duplicate workgroup variable name %q", w.Name)
		}
		wgnames[w.Name] = true
		if !types.IsWorkgroupStorable(w.Type) {
			return fmt.Errorf("workgroup variable %s has invalid type %s", w.Name, w.Type)
		}
	}
	e := verifyEnv{values: map[ValueID]*types.Type{}, places: map[PlaceID]placeInfo{}}
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
	_, err := verifyBlock(m, f, f.Body, e, fmap, "return")
	if err != nil {
		return err
	}
	if f.Compute {
		if err := verifyUniformity(m, f, fmap); err != nil {
			return err
		}
	}
	return nil
}

func verifyBlock(m *Module, f *Function, b *Block, e verifyEnv, fmap map[string]*Function, termKind string) (verifyEnv, error) {
	if b == nil {
		return e, fmt.Errorf("nil block")
	}
	defVal := func(id ValueID, t *types.Type) error {
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
	val := func(id ValueID) (*types.Type, error) {
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
			if !types.IsScalar(x.Type) {
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
			if !types.Equal(t, x.Type) {
				return e, fmt.Errorf("unary %s type mismatch %s -> %s", x.Op, t, x.Type)
			}
			if x.Op == "!" && !types.Equal(t, types.TBool) {
				return e, fmt.Errorf("! requires boolean")
			}
			if x.Op == "-" && !types.IsSignedNumeric(t) {
				return e, fmt.Errorf("unary - requires signed/float numeric")
			}
			if x.Op == "~" && !types.IsIntegerLike(t) {
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
			if !types.Equal(t, x.From) {
				return e, fmt.Errorf("convert source says %s but value is %s", x.From, t)
			}
			if !types.IsNumericScalar(t) || !types.IsNumericScalar(x.Type) {
				return e, fmt.Errorf("convert supports scalar numerics")
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Composite:
			if x.Type.Kind == types.Struct {
				if len(x.Values) != len(x.Type.Fields) {
					return e, fmt.Errorf("struct %s construction has %d values", x.Type, len(x.Values))
				}
				for i, id := range x.Values {
					t, err := val(id)
					if err != nil {
						return e, err
					}
					if !types.Equal(t, x.Type.Fields[i].Type) {
						return e, fmt.Errorf("struct %s field %s has %s, want %s", x.Type, x.Type.Fields[i].Name, t, x.Type.Fields[i].Type)
					}
				}
			} else if x.Type.Kind == types.Vector {
				if len(x.Values) != x.Type.Lanes {
					return e, fmt.Errorf("vector construction has %d components, want %d", len(x.Values), x.Type.Lanes)
				}
				for _, id := range x.Values {
					t, err := val(id)
					if err != nil {
						return e, err
					}
					if !types.Equal(t, x.Type.Elem) {
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
			var et *types.Type
			if bt.Kind == types.Struct {
				if x.Index < 0 || x.Index >= len(bt.Fields) {
					return e, fmt.Errorf("struct extract index %d out of range", x.Index)
				}
				et = bt.Fields[x.Index].Type
			} else if bt.Kind == types.Vector {
				if x.Index < 0 || x.Index >= bt.Lanes {
					return e, fmt.Errorf("vector extract index %d out of range", x.Index)
				}
				et = bt.Elem
			} else {
				return e, fmt.Errorf("extract from %s", bt)
			}
			if !types.Equal(et, x.Type) {
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
			if bt.Kind != types.Vector {
				return e, fmt.Errorf("vector index base is %s", bt)
			}
			if !types.IsInteger(it) {
				return e, fmt.Errorf("vector index is %s, want i32 or u32", it)
			}
			if !types.Equal(x.Type, bt.Elem) {
				return e, fmt.Errorf("vector index type %s, want %s", x.Type, bt.Elem)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *Intrinsic:
			args := make([]*types.Type, len(x.Args))
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
			if callee.Compute {
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
				if !types.Equal(t, callee.Params[i].Type) {
					return e, fmt.Errorf("call %s arg %d is %s, want %s", x.Function, i, t, callee.Params[i].Type)
				}
			}
			if !types.Equal(x.Type, callee.Return) {
				return e, fmt.Errorf("call %s result says %s, want %s", x.Function, x.Type, callee.Return)
			}
			if x.Type.Kind != types.Void {
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			}
		case *PlaceRoot:
			if x.Resource < 0 || x.Resource >= len(m.Resources) {
				return e, fmt.Errorf("invalid resource %d", x.Resource)
			}
			r := m.Resources[x.Resource]
			if !types.Equal(x.Type, r.Type) {
				return e, fmt.Errorf("resource place type %s, want %s", x.Type, r.Type)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, x.Resource}); err != nil {
				return e, err
			}
		case *PlaceWorkgroup:
			if !f.Compute || x.Workgroup < 0 || x.Workgroup >= len(f.WorkgroupVars) {
				return e, fmt.Errorf("invalid workgroup place %d", x.Workgroup)
			}
			w := f.WorkgroupVars[x.Workgroup]
			if !types.Equal(x.Type, w.Type) {
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
			if bp.ty.Kind != types.Struct || x.Field < 0 || x.Field >= len(bp.ty.Fields) {
				return e, fmt.Errorf("invalid field place on %s", bp.ty)
			}
			want := bp.ty.Fields[x.Field].Type
			if !types.Equal(want, x.Type) {
				return e, fmt.Errorf("field place type %s, want %s", x.Type, want)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, bp.resource}); err != nil {
				return e, err
			}
		case *PlaceIndex:
			bp, err := place(x.Base)
			if err != nil {
				return e, err
			}
			if bp.ty.Kind != types.RuntimeArray && bp.ty.Kind != types.FixedArray && bp.ty.Kind != types.Vector {
				return e, fmt.Errorf("index place base is %s", bp.ty)
			}
			it, err := val(x.Index)
			if err != nil {
				return e, err
			}
			if !types.Equal(it, types.TU32) && !types.Equal(it, types.TI32) {
				return e, fmt.Errorf("array index is %s", it)
			}
			if !types.Equal(x.Type, bp.ty.Elem) {
				return e, fmt.Errorf("index result %s, want %s", x.Type, bp.ty.Elem)
			}
			if err := defPlace(x.Result, placeInfo{x.Type, bp.resource}); err != nil {
				return e, err
			}
		case *Load:
			p, err := place(x.Place)
			if err != nil {
				return e, err
			}
			if !types.IsConstructible(p.ty) {
				return e, fmt.Errorf("place of type %s cannot be loaded as a value", p.ty)
			}
			if !types.Equal(p.ty, x.Type) {
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
			if !types.Equal(p.ty, v) {
				return e, fmt.Errorf("store %s into %s", v, p.ty)
			}
			if !types.IsConstructible(p.ty) {
				return e, fmt.Errorf("place of type %s cannot be stored as a whole value", p.ty)
			}
			if p.resource >= 0 {
				r := m.Resources[p.resource]
				if r.Kind != Buffer || r.Access != Mutable {
					return e, fmt.Errorf("store through non-writable resource %s", r.Name)
				}
			}
		case *Atomic:
			p, err := place(x.Place)
			if err != nil {
				return e, err
			}
			if p.ty.Kind != types.Atomic || !types.Equal(p.ty.Elem, x.Type) || (x.Type.Kind != types.I32 && x.Type.Kind != types.U32) {
				return e, fmt.Errorf("atomic operation type %s does not match place %s", x.Type, p.ty)
			}
			if p.resource >= 0 && x.Op != AtomicLoad {
				r := m.Resources[p.resource]
				if r.Kind != Buffer || r.Access != Mutable {
					return e, fmt.Errorf("atomic operation through non-writable buffer resource %s", r.Name)
				}
			}
			switch x.Op {
			case AtomicLoad:
				if x.Result == 0 || x.Value != 0 {
					return e, fmt.Errorf("atomicLoad result/value shape is invalid")
				}
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			case AtomicStore:
				if x.Result != 0 || x.Value == 0 {
					return e, fmt.Errorf("atomicStore result/value shape is invalid")
				}
				vt, err := val(x.Value)
				if err != nil || !types.Equal(vt, x.Type) {
					if err != nil {
						return e, err
					}
					return e, fmt.Errorf("atomicStore value is %s, want %s", vt, x.Type)
				}
			case AtomicAdd, AtomicSub, AtomicMin, AtomicMax, AtomicAnd, AtomicOr, AtomicXor, AtomicExchange:
				if x.Result == 0 || x.Value == 0 {
					return e, fmt.Errorf("atomic read-modify-write result/value shape is invalid")
				}
				vt, err := val(x.Value)
				if err != nil {
					return e, err
				}
				if !types.Equal(vt, x.Type) {
					return e, fmt.Errorf("atomic operand is %s, want %s", vt, x.Type)
				}
				if err := defVal(x.Result, x.Type); err != nil {
					return e, err
				}
			default:
				return e, fmt.Errorf("unknown atomic operation %d", x.Op)
			}
		case *Barrier:
			if !f.Compute {
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
			if p.ty.Kind != types.RuntimeArray {
				return e, fmt.Errorf("array_length on %s", p.ty)
			}
			if !types.Equal(x.Type, types.TU32) {
				return e, fmt.Errorf("array_length result must be u32")
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return e, err
			}
		case *If:
			ct, err := val(x.Cond)
			if err != nil {
				return e, err
			}
			if !types.Equal(ct, types.TBool) {
				return e, fmt.Errorf("if condition is %s", ct)
			}
			te, err := verifyBlock(m, f, x.Then, e.clone(), fmap, "yield")
			if err != nil {
				return e, fmt.Errorf("if then: %w", err)
			}
			ee, err := verifyBlock(m, f, x.Else, e.clone(), fmap, "yield")
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
					if !types.Equal(a, r.Type) {
						return e, fmt.Errorf("if then result %d type mismatch", i)
					}
				}
				if ok2 {
					bb := ee.values[ey.Values[i]]
					if !types.Equal(bb, r.Type) {
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
				if !types.Equal(it, p.Type) {
					return e, fmt.Errorf("loop init %s, want %s", it, p.Type)
				}
				if _, exists := le.values[p.ID]; exists {
					return e, fmt.Errorf("loop param %%%d redefined", p.ID)
				}
				le.values[p.ID] = p.Type
			}
			ce, err := verifyBlock(m, f, x.Cond, le.clone(), fmap, "yield")
			if err != nil {
				return e, fmt.Errorf("loop condition: %w", err)
			}
			cy, ok := x.Cond.Term.(*Yield)
			if !ok || len(cy.Values) != 1 || !types.Equal(ce.values[cy.Values[0]], types.TBool) {
				return e, fmt.Errorf("loop condition must yield one boolean")
			}
			be, err := verifyBlock(m, f, x.Body, le.clone(), fmap, "continue")
			if err != nil {
				return e, fmt.Errorf("loop body: %w", err)
			}
			co, ok := x.Body.Term.(*Continue)
			if !ok {
				return e, fmt.Errorf("loop body must continue")
			}
			if len(co.Values) != len(x.Params) || len(x.Results) != len(x.Params) {
				return e, fmt.Errorf("loop carried arity mismatch")
			}
			for i, p := range x.Params {
				t := be.values[co.Values[i]]
				if !types.Equal(t, p.Type) || !types.Equal(x.Results[i].Type, p.Type) {
					return e, fmt.Errorf("loop carried value %d type mismatch", i)
				}
				if err := defVal(x.Results[i].ID, p.Type); err != nil {
					return e, err
				}
			}
		default:
			return e, fmt.Errorf("unknown instruction %T", in)
		}
	}
	if b.Term == nil {
		return e, fmt.Errorf("block has no terminator")
	}
	switch t := b.Term.(type) {
	case *Return:
		if termKind != "return" && termKind != "yield" && termKind != "continue" {
		} // returns are legal nested control exits.
		if f.Return.Kind == types.Void {
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
			if !types.Equal(vt, f.Return) {
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
		if termKind != "continue" {
			return e, fmt.Errorf("unexpected continue terminator")
		}
		for _, id := range t.Values {
			if _, err := val(id); err != nil {
				return e, err
			}
		}
	case *Unreachable:
		// Structured constructs whose every path exits can leave an unreachable merge.
	default:
		return e, fmt.Errorf("unknown terminator %T", b.Term)
	}
	return e, nil
}

func verifyIntrinsic(x *Intrinsic, args []*types.Type) error {
	need := func(n int) error {
		if len(args) != n {
			return fmt.Errorf("intrinsic %s has %d args, want %d", x.Kind, len(args), n)
		}
		return nil
	}
	same := func() bool {
		for _, t := range args {
			if !types.Equal(t, args[0]) {
				return false
			}
		}
		return len(args) > 0
	}
	floatVec := func(t *types.Type) bool {
		return t != nil && t.Kind == types.Vector && t.Elem != nil && t.Elem.Kind == types.F32
	}
	switch x.Kind {
	case IntrinsicAbs:
		if err := need(1); err != nil {
			return err
		}
		t := args[0]
		baseOK := t.Kind == types.I32 || t.Kind == types.F32 || t.Kind == types.Vector && (t.Elem.Kind == types.I32 || t.Elem.Kind == types.F32)
		if !baseOK || !types.Equal(x.Type, t) {
			return fmt.Errorf("abs requires i32/f32 scalar or vector and preserves type")
		}
	case IntrinsicFloor, IntrinsicCeil, IntrinsicTrunc, IntrinsicSin, IntrinsicCos, IntrinsicTan, IntrinsicExp, IntrinsicExp2, IntrinsicLog, IntrinsicLog2, IntrinsicSqrt, IntrinsicRSqrt:
		if err := need(1); err != nil {
			return err
		}
		if !types.IsFloatLike(args[0]) || !types.Equal(x.Type, args[0]) {
			return fmt.Errorf("intrinsic %s requires f32 scalar/vector and preserves type", x.Kind)
		}
	case IntrinsicPow:
		if err := need(2); err != nil {
			return err
		}
		if !same() || !types.IsFloatLike(args[0]) || !types.Equal(x.Type, args[0]) {
			return fmt.Errorf("pow requires matching f32 scalar/vector operands")
		}
	case IntrinsicMin, IntrinsicMax:
		if err := need(2); err != nil {
			return err
		}
		if !same() || !types.IsIntegerLike(args[0]) || !types.Equal(x.Type, args[0]) {
			return fmt.Errorf("%s requires matching integer scalar/vector operands", x.Kind)
		}
	case IntrinsicClamp:
		if err := need(3); err != nil {
			return err
		}
		if !same() || !types.IsIntegerLike(args[0]) || !types.Equal(x.Type, args[0]) {
			return fmt.Errorf("clamp requires three matching integer scalar/vector operands")
		}
	case IntrinsicDot:
		if err := need(2); err != nil {
			return err
		}
		if !same() || !floatVec(args[0]) || !types.Equal(x.Type, types.TF32) {
			return fmt.Errorf("dot requires matching f32 vectors and returns f32")
		}
	case IntrinsicLength:
		if err := need(1); err != nil {
			return err
		}
		if !floatVec(args[0]) || !types.Equal(x.Type, types.TF32) {
			return fmt.Errorf("length requires an f32 vector and returns f32")
		}
	case IntrinsicDistance:
		if err := need(2); err != nil {
			return err
		}
		if !same() || !floatVec(args[0]) || !types.Equal(x.Type, types.TF32) {
			return fmt.Errorf("distance requires matching f32 vectors and returns f32")
		}
	case IntrinsicCross:
		if err := need(2); err != nil {
			return err
		}
		if !same() || !floatVec(args[0]) || args[0].Lanes != 3 || !types.Equal(x.Type, args[0]) {
			return fmt.Errorf("cross requires two f32x3 operands")
		}
	case IntrinsicNormalize:
		if err := need(1); err != nil {
			return err
		}
		if !floatVec(args[0]) || !types.Equal(x.Type, args[0]) {
			return fmt.Errorf("normalize requires an f32 vector and preserves type")
		}
	default:
		return fmt.Errorf("unknown intrinsic %d", x.Kind)
	}
	return nil
}

func verifyBinary(x *Binary, l, r *types.Type) error {
	switch x.Op {
	case "+", "-":
		if !types.Equal(l, r) || !types.Equal(x.Type, l) || !types.IsNumeric(l) {
			return fmt.Errorf("%s requires matching numeric operands; got %s and %s -> %s", x.Op, l, r, x.Type)
		}
	case "*":
		if types.Equal(l, r) && types.Equal(x.Type, l) && types.IsNumeric(l) {
			return nil
		}
		if l.Kind == types.Vector && types.Equal(r, l.Elem) && types.Equal(x.Type, l) {
			return nil
		}
		if r.Kind == types.Vector && types.Equal(l, r.Elem) && types.Equal(x.Type, r) {
			return nil
		}
		return fmt.Errorf("* invalid for %s and %s -> %s", l, r, x.Type)
	case "/", "%":
		if types.Equal(l, r) && types.Equal(x.Type, l) && types.IsNumeric(l) {
			return nil
		}
		if x.Op == "/" && l.Kind == types.Vector && types.Equal(r, l.Elem) && types.Equal(x.Type, l) {
			return nil
		}
		return fmt.Errorf("%s invalid for %s and %s", x.Op, l, r)
	case "==", "!=", "<", "<=", ">", ">=":
		if !types.Equal(l, r) || !types.IsNumericScalar(l) || !types.Equal(x.Type, types.TBool) {
			return fmt.Errorf("comparison invalid for %s and %s", l, r)
		}
	case "&&", "||":
		if !types.Equal(l, types.TBool) || !types.Equal(r, types.TBool) || !types.Equal(x.Type, types.TBool) {
			return fmt.Errorf("logical op requires booleans")
		}
	case "&", "|", "^":
		if !types.Equal(l, r) || !types.Equal(x.Type, l) || !types.IsIntegerLike(l) {
			return fmt.Errorf("%s requires matching integer scalar/vector operands; got %s and %s -> %s", x.Op, l, r, x.Type)
		}
	case "<<", ">>":
		want := types.ShiftCountType(l)
		if want == nil || !types.Equal(r, want) || !types.Equal(x.Type, l) {
			return fmt.Errorf("%s requires integer value %s shifted by %s; got %s and %s -> %s", x.Op, l, want, l, r, x.Type)
		}
	default:
		return fmt.Errorf("unknown binary op %q", x.Op)
	}
	return nil
}

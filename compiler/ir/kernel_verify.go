package ir

import (
	"fmt"
	"maps"
	"tach/foundation"
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

func VerifyKernel(m *KernelModule) error {
	if m == nil {
		return fmt.Errorf("nil IR module")
	}
	names := map[string]bool{}
	for index, s := range m.Structs {
		if s == nil {
			return fmt.Errorf("module struct %d is nil", index)
		}
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
	for index, f := range m.Functions {
		if f == nil {
			return fmt.Errorf("function %d is nil", index)
		}
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

func verifyFunction(m *KernelModule, f *Function, fmap map[string]*Function) error {
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

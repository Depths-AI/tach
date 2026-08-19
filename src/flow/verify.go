package flow

import (
	"fmt"

	"tach/src/ir"
	"tach/src/types"
)

func Verify(m *Module) error {
	if m == nil || m.Kernel == nil {
		return fmt.Errorf("flow module is missing Kernel IR")
	}
	if err := ir.Verify(m.Kernel); err != nil {
		return fmt.Errorf("kernel IR: %w", err)
	}
	names := map[string]bool{}
	for i, p := range m.Programs {
		if p == nil {
			return fmt.Errorf("program %d is nil", i)
		}
		if p.Name == "" || names[p.Name] {
			return fmt.Errorf("program %d has missing or duplicate name %q", i, p.Name)
		}
		names[p.Name] = true
		if err := verifyProgram(m, p); err != nil {
			return fmt.Errorf("program %s: %w", p.Name, err)
		}
	}
	return nil
}

func verifyProgram(m *Module, p *Program) error {
	if len(p.Dispatches) == 0 {
		return fmt.Errorf("has no dispatches")
	}
	bufferCount := 0
	for i, parameter := range p.Parameters {
		if parameter.Name == "" || parameter.Type == nil {
			return fmt.Errorf("parameter %d is incomplete", i)
		}
		if parameter.Kind == BufferParameter {
			bufferCount++
			r := p.Resource(parameter.Resource)
			if r == nil || r.Kind != External || r.Parameter != i || !types.Equal(r.Type, parameter.Type) {
				return fmt.Errorf("buffer parameter %d has invalid resource", i)
			}
		} else if parameter.Kind != ValueParameter || !types.IsConstructible(parameter.Type) {
			return fmt.Errorf("parameter %d has invalid kind/type", i)
		}
	}
	if bufferCount == 0 && p.View == nil {
		return fmt.Errorf("requires an external buffer")
	}
	for i, r := range p.Resources {
		if r.ID != ResourceID(i+1) || r.Type == nil {
			return fmt.Errorf("resource IDs must be dense from one")
		}
		if r.Kind != External && r.Kind != Transient {
			return fmt.Errorf("resource %d has invalid kind", r.ID)
		}
		if p.Version(r.Initial) == nil || p.Version(r.Final) == nil {
			return fmt.Errorf("resource %d has invalid initial/final version", r.ID)
		}
		if r.Kind == Transient && p.Shape(r.Length) == nil {
			return fmt.Errorf("transient %d has invalid length", r.ID)
		}
	}
	producers := map[VersionID]DispatchID{}
	for i, v := range p.Versions {
		if v.ID != VersionID(i+1) || p.Resource(v.Resource) == nil {
			return fmt.Errorf("version IDs must be dense and reference a resource")
		}
		if v.Previous != 0 {
			previous := p.Version(v.Previous)
			if previous == nil || previous.Resource != v.Resource || previous.ID >= v.ID {
				return fmt.Errorf("version %d has invalid predecessor", v.ID)
			}
		}
		if v.Producer != 0 {
			producers[v.ID] = v.Producer
		}
	}
	for i, s := range p.Shapes {
		if s.ID != ShapeID(i+1) {
			return fmt.Errorf("shape IDs must be dense from one")
		}
		if err := verifyShape(p, s.ID, map[ShapeID]bool{}); err != nil {
			return err
		}
	}
	current := map[ResourceID]VersionID{}
	for _, r := range p.Resources {
		current[r.ID] = r.Initial
	}
	if p.View != nil {
		view, resource := p.View, p.Resource(p.View.Source)
		pixel := types.Vec(types.TF32, 4)
		if view.Format != SRGB8 || resource == nil || resource.Type.Kind != types.RuntimeArray || !types.Equal(resource.Type.Elem, pixel) {
			return fmt.Errorf("has invalid view source/format")
		}
		if p.Version(view.Input) == nil || p.Version(view.Input).Resource != view.Source || p.Shape(view.Width) == nil || p.Shape(view.Height) == nil {
			return fmt.Errorf("has invalid view version/extent")
		}
	}
	for i, d := range p.Dispatches {
		if d.ID != DispatchID(i+1) {
			return fmt.Errorf("dispatch IDs must be dense from one")
		}
		stage := m.Kernel.Function(d.Stage)
		if stage == nil || stage.Kind != ir.Stage {
			return fmt.Errorf("dispatch %d references non-stage %q", d.ID, d.Stage)
		}
		if len(d.Domain) != len(stage.Indices) || len(d.Domain) < 1 || len(d.Domain) > 3 {
			return fmt.Errorf("dispatch %d domain rank mismatch", d.ID)
		}
		for _, shape := range d.Domain {
			if p.Shape(shape) == nil {
				return fmt.Errorf("dispatch %d has invalid domain", d.ID)
			}
		}
		if len(d.Buffers) != len(stage.BufferParams) || len(d.Values) != len(stage.Params) {
			return fmt.Errorf("dispatch %d argument count mismatch", d.ID)
		}
		seen := map[ResourceID]bool{}
		summary := ir.AnalyzeAccess(stage)
		for formal, a := range d.Buffers {
			if a.Formal != formal || p.Resource(a.Resource) == nil || seen[a.Resource] || current[a.Resource] != a.Input {
				return fmt.Errorf("dispatch %d has invalid buffer argument %d", d.ID, formal)
			}
			seen[a.Resource] = true
			if !types.Equal(stage.BufferParams[formal].Type, p.Resource(a.Resource).Type) {
				return fmt.Errorf("dispatch %d buffer type mismatch", d.ID)
			}
			input := p.Version(a.Input)
			if input == nil || !input.Defined && summary.Buffers[formal].Read {
				return fmt.Errorf("dispatch %d reads undefined resource %d", d.ID, a.Resource)
			}
			if stage.BufferParams[formal].Access == ir.Mutable {
				v := p.Version(a.Output)
				if v == nil || v.Resource != a.Resource || v.Previous != a.Input || v.Producer != d.ID {
					return fmt.Errorf("dispatch %d has invalid output version", d.ID)
				}
				if v.Defined != (input.Defined || summary.Buffers[formal].CompleteWrite) {
					return fmt.Errorf("dispatch %d output definition proof mismatch", d.ID)
				}
				current[a.Resource] = a.Output
			} else if a.Output != 0 {
				return fmt.Errorf("dispatch %d read-only buffer has output", d.ID)
			}
		}
		for formal, a := range d.Values {
			if a.Formal != formal || !validValue(p, a, stage.Params[formal].Type) {
				return fmt.Errorf("dispatch %d has invalid value argument %d", d.ID, formal)
			}
		}
	}
	for _, r := range p.Resources {
		if current[r.ID] != r.Final {
			return fmt.Errorf("resource %d final version mismatch", r.ID)
		}
	}
	if p.View != nil {
		if current[p.View.Source] != p.View.Input || !p.Version(p.View.Input).Defined {
			return fmt.Errorf("view source is not the defined final resource version")
		}
	}
	_ = producers
	return nil
}

func verifyShape(p *Program, id ShapeID, active map[ShapeID]bool) error {
	if active[id] {
		return fmt.Errorf("shape %d is cyclic", id)
	}
	s := p.Shape(id)
	if s == nil {
		return fmt.Errorf("invalid shape %d", id)
	}
	active[id] = true
	defer delete(active, id)
	switch s.Op {
	case ShapeConstant:
		return nil
	case ShapeParameter:
		if s.Parameter < 0 || s.Parameter >= len(p.Parameters) || p.Parameters[s.Parameter].Kind != ValueParameter || !types.Equal(pathType(p.Parameters[s.Parameter].Type, s.Path), types.TU32) {
			return fmt.Errorf("shape %d has invalid parameter", id)
		}
		return nil
	case ShapeResourceLength:
		resource := p.Resource(s.Resource)
		if resource == nil {
			return fmt.Errorf("shape %d has invalid resource", id)
		}
		final := pathType(resource.Type, s.Path)
		if final == nil || final.Kind != types.RuntimeArray {
			return fmt.Errorf("shape %d resource path is not a runtime array", id)
		}
		return nil
	case ShapeLaunchAxis:
		if !p.Indexed || int(s.Axis) >= p.Rank {
			return fmt.Errorf("shape %d has invalid launch axis", id)
		}
		return nil
	case ShapeAdd, ShapeSub, ShapeMul, ShapeDiv, ShapeRem, ShapeMin, ShapeMax, ShapeCeilDiv:
		if err := verifyShape(p, s.Left, active); err != nil {
			return err
		}
		return verifyShape(p, s.Right, active)
	default:
		return fmt.Errorf("shape %d has invalid operation", id)
	}
}

func validValue(p *Program, value ValueArgument, want *types.Type) bool {
	switch value.Kind {
	case ValueParameterRef:
		return value.Parameter >= 0 && value.Parameter < len(p.Parameters) && p.Parameters[value.Parameter].Kind == ValueParameter && types.Equal(pathType(p.Parameters[value.Parameter].Type, value.Path), want)
	case ValueBool:
		return want.Kind == types.Bool && value.Bits <= 1
	case ValueI32:
		return want.Kind == types.I32
	case ValueU32, ValueRepeat:
		return want.Kind == types.U32
	case ValueF16Bits:
		return want.Kind == types.F16 && value.Bits <= 0xffff
	case ValueF32Bits:
		return want.Kind == types.F32
	case ValueShape:
		return want.Kind == types.U32 && p.Shape(value.Shape) != nil
	default:
		return false
	}
}

func pathType(root *types.Type, path []string) *types.Type {
	current := root
	for _, name := range path {
		if current == nil || current.Kind != types.Struct {
			return nil
		}
		index := types.FieldIndex(current, name)
		if index < 0 {
			return nil
		}
		current = current.Fields[index].Type
	}
	return current
}

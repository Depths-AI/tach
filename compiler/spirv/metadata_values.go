package spirv

import (
	"fmt"

	"tach/foundation"
	"tach/ir"
)

func describeShape(program *ir.Program, id ir.ShapeID) (shapeExpression, error) {
	shape := program.Shape(id)
	if shape == nil {
		return shapeExpression{}, fmt.Errorf("invalid shape %d", id)
	}
	out := shapeExpression{Path: append([]string(nil), shape.Path...)}
	if shape.Op == ir.ShapeConstant {
		out.Op, out.Value = "constant", shape.Value
		return out, nil
	}
	if shape.Op == ir.ShapeParameter {
		out.Op, out.Parameter = "parameter", shape.Parameter
		return out, nil
	}
	if shape.Op == ir.ShapeResourceLength {
		out.Op, out.Resource = "resourceLength", externalResourceIndex(program, shape.Resource)
		return out, nil
	}
	if shape.Op == ir.ShapeLaunchAxis {
		out.Op, out.Axis = "launchAxis", shape.Axis
		return out, nil
	}
	out.Op = shape.Op.String()
	left, err := describeShape(program, shape.Left)
	if err != nil {
		return out, err
	}
	right, err := describeShape(program, shape.Right)
	if err != nil {
		return out, err
	}
	out.Left, out.Right = &left, &right
	return out, nil
}

func externalResourceIndex(program *ir.Program, id ir.ResourceID) int {
	index := 0
	for _, resource := range program.Resources {
		if resource.Kind == ir.ExternalResourceKind {
			if resource.ID == id {
				return index
			}
			index++
		}
	}
	return -1
}

func describeValue(program *ir.Program, argument ir.ValueArgument, fieldPath []string) (valueSource, error) {
	if argument.Kind == ir.ValueFromParameter {
		return valueSource{Kind: "parameter", Parameter: argument.Parameter, Path: append(append([]string(nil), argument.Path...), fieldPath...)}, nil
	}
	if argument.Kind == ir.ValueFromShape {
		expression, err := describeShape(program, argument.Shape)
		return valueSource{Kind: "shape", Expression: &expression}, err
	}
	if argument.Kind == ir.ValueFromRepeat {
		return valueSource{Kind: "repeat"}, nil
	}
	if argument.Kind == ir.ValueFromConstant {
		return valueSource{}, fmt.Errorf("compile-time constant reached runtime metadata")
	}
	return valueSource{}, fmt.Errorf("invalid value source")
}

func describeParameterBlock(block *ir.HostParameterBlock) (*parameterBlockMetadata, error) {
	if block == nil {
		return nil, nil
	}
	out := &parameterBlockMetadata{Group: 0, Binding: block.Binding, ByteSize: block.Layout.Size, Fields: make([]parameterFieldMetadata, len(block.Fields))}
	for i, field := range block.Fields {
		layout, err := describeParameterLayout(field.Logical)
		if err != nil {
			return nil, err
		}
		out.Fields[i] = parameterFieldMetadata{Type: field.Logical.String(), ByteOffset: field.Offset, Layout: layout}
	}
	return out, nil
}

func describeDomain(program *ir.Program, ids []ir.ShapeID) ([]shapeExpression, error) {
	out := make([]shapeExpression, 0, len(ids))
	for _, id := range ids {
		expression, err := describeShape(program, id)
		if err != nil {
			return nil, err
		}
		out = append(out, expression)
	}
	return out, nil
}

func describeParameterLayout(t *foundation.Type) (*hostLayout, error) {
	if t.Kind == foundation.AtomicKind {
		return describeParameterLayout(t.Elem)
	}
	if t.Kind == foundation.BoolKind {
		return &hostLayout{Kind: "bool", Size: 4}, nil
	}
	l, err := foundation.LayoutOf(t)
	if err != nil {
		return nil, err
	}
	out := &hostLayout{Size: l.Size, Stride: l.Stride, Runtime: l.Runtime}
	if name := map[foundation.TypeKind]string{
		foundation.Int32Kind: "i32", foundation.Uint32Kind: "u32",
		foundation.Float16Kind: "f16", foundation.Float32Kind: "f32",
	}[t.Kind]; name != "" {
		out.Kind = name
		return out, nil
	}
	switch t.Kind {
	case foundation.VectorKind:
		out.Kind, out.Count = "vector", uint32(t.Lanes)
		out.Elem, err = describeParameterLayout(t.Elem)
	case foundation.FixedArrayKind:
		out.Kind, out.Count = "array", t.Count
		out.Elem, err = describeParameterLayout(t.Elem)
	case foundation.RuntimeArrayKind:
		out.Kind = "runtime"
		out.Elem, err = describeParameterLayout(t.Elem)
	case foundation.StructKind:
		out.Kind, out.Fields = "struct", []hostLayoutField{}
		for i, field := range t.Fields {
			child, childErr := describeParameterLayout(field.Type)
			if childErr != nil {
				return nil, childErr
			}
			out.Fields = append(out.Fields, hostLayoutField{Name: field.Name, Offset: l.Fields[i].Offset, Type: child})
		}
	default:
		return nil, fmt.Errorf("cannot describe host type %s", t)
	}
	return out, err
}

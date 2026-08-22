package wgsl

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
	switch shape.Op {
	case ir.ShapeConstant:
		out.Op, out.Value = "constant", shape.Value
	case ir.ShapeParameter:
		out.Op, out.Parameter = "parameter", shape.Parameter
	case ir.ShapeResourceLength:
		out.Op, out.Resource = "resourceLength", externalResourceIndex(program, shape.Resource)
	case ir.ShapeLaunchAxis:
		out.Op, out.Axis = "launchAxis", shape.Axis
	default:
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
	}
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
	switch argument.Kind {
	case ir.ValueFromParameter:
		return valueSource{Kind: "parameter", Parameter: argument.Parameter, Path: append(append([]string(nil), argument.Path...), fieldPath...)}, nil
	case ir.ValueFromShape:
		expression, err := describeShape(program, argument.Shape)
		return valueSource{Kind: "shape", Expression: &expression}, err
	case ir.ValueFromRepeat:
		return valueSource{Kind: "repeat"}, nil
	case ir.ValueFromConstant:
		return valueSource{}, fmt.Errorf("compile-time constant reached runtime metadata")
	default:
		return valueSource{}, fmt.Errorf("invalid value source")
	}
}

func describeParameterBlock(block *ir.HostParameterBlock) (*parameterBlockMetadata, error) {
	if block == nil {
		return nil, nil
	}
	out := &parameterBlockMetadata{Group: 0, Binding: block.Binding, ByteSize: block.Layout.Size, Fields: []parameterFieldMetadata{}}
	for _, field := range block.Fields {
		layout, err := describeParameterLayout(field.Logical)
		if err != nil {
			return nil, err
		}
		out.Fields = append(out.Fields, parameterFieldMetadata{Type: field.Logical.String(), ByteOffset: field.Offset, Layout: layout})
	}
	return out, nil
}

func describeTransient(program *ir.Program, t *foundation.Type, stride, alignment, minimum uint32, lengthID ir.ShapeID, color, first, last int) (transientMetadata, error) {
	length, err := describeShape(program, lengthID)
	return transientMetadata{Type: t.String(), Stride: stride, Alignment: alignment, MinimumByteSize: minimum, Length: length, Color: color, FirstStep: first, LastStep: last}, err
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

func describeParameters(program *ir.Program, block *ir.HostParameterBlock, arguments []ir.ValueArgument) ([]valueSource, error) {
	if block == nil {
		return []valueSource{}, nil
	}
	out := make([]valueSource, 0, len(block.Fields))
	for _, field := range block.Fields {
		if field.Parameter < 0 || field.Parameter >= len(arguments) {
			return nil, fmt.Errorf("kernel parameter field references missing dispatch value")
		}
		source, err := describeValue(program, arguments[field.Parameter], field.Path)
		if err != nil {
			return nil, err
		}
		out = append(out, source)
	}
	return out, nil
}

func describeView(program *ir.Program, step stepMetadata, widthID, heightID ir.ShapeID, outputColor int, output uint32, fused bool) (*viewMetadata, error) {
	width, err := describeShape(program, widthID)
	if err != nil {
		return nil, err
	}
	height, err := describeShape(program, heightID)
	if err != nil {
		return nil, err
	}
	return &viewMetadata{Format: "srgb8", Step: step, Width: width, Height: height, OutputColor: outputColor, Output: output, Fused: fused}, nil
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
	switch t.Kind {
	case foundation.Int32Kind:
		out.Kind = "i32"
	case foundation.Uint32Kind:
		out.Kind = "u32"
	case foundation.Float16Kind:
		out.Kind = "f16"
	case foundation.Float32Kind:
		out.Kind = "f32"
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

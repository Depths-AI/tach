package driver

import (
	"encoding/json"
	"fmt"
	"tach/foundation"
	"tach/ir"
)

func generateMetadata(logical *ir.Module, web, spirv []byte) (*Metadata, []byte, error) {
	if logical == nil || !json.Valid(web) || !json.Valid(spirv) {
		return nil, nil, fmt.Errorf("incomplete compiled program")
	}
	if err := ir.Verify(logical); err != nil {
		return nil, nil, err
	}
	metadata, err := buildMetadata(logical, web, spirv)
	if err != nil {
		return nil, nil, err
	}
	metadataJSON, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return nil, nil, err
	}
	return metadata, append(metadataJSON, '\n'), nil
}

func buildMetadata(logical *ir.Module, web, spirv json.RawMessage) (*Metadata, error) {
	metadata := &Metadata{Schema: 2, Types: []TypeMetadata{}, Programs: []PublicProgramMeta{}}
	for _, t := range logical.Kernel.Structs {
		item := TypeMetadata{Name: t.Name, Fields: []FieldMeta{}}
		for _, field := range t.Fields {
			item.Fields = append(item.Fields, FieldMeta{Name: field.Name, Type: field.Type.String()})
		}
		metadata.Types = append(metadata.Types, item)
	}
	for _, program := range logical.Programs {
		item := PublicProgramMeta{Name: program.Name, Parameters: []PublicParameterMeta{}, Resources: []ExternalResourceMeta{}, View: program.View != nil}
		external := map[ir.ResourceID]int{}
		for _, resource := range program.Resources {
			if resource.Kind != ir.ExternalResourceKind {
				continue
			}
			index := len(item.Resources)
			external[resource.ID] = index
			description, err := externalResource(resource)
			if err != nil {
				return nil, err
			}
			item.Resources = append(item.Resources, description)
		}
		for _, parameter := range program.Parameters {
			p := PublicParameterMeta{Name: parameter.Name, Type: parameter.Type.String()}
			if parameter.Kind == ir.BufferParameterKind {
				p.Kind = "buffer"
				index := external[parameter.Resource]
				p.Resource = &index
			} else {
				p.Kind = "value"
			}
			item.Parameters = append(item.Parameters, p)
		}
		if program.Indexed {
			item.Launch = &LaunchMeta{Dimensions: program.Rank}
			if program.Rank == 1 {
				for i, resource := range item.Resources {
					if resource.Runtime {
						index := i
						item.Launch.InferFromResource = &index
						break
					}
				}
			}
		}
		metadata.Programs = append(metadata.Programs, item)
	}
	metadata.Targets.Web, metadata.Targets.SPIRV = web, spirv
	return metadata, nil
}

func externalResource(resource ir.Resource) (ExternalResourceMeta, error) {
	l, err := foundation.LayoutOf(resource.Type)
	if err != nil {
		return ExternalResourceMeta{}, err
	}
	host, err := describeHostLayout(resource.Type)
	if err != nil {
		return ExternalResourceMeta{}, err
	}
	out := ExternalResourceMeta{Name: resource.Name, Type: resource.Type.String(), Alignment: l.Align, Runtime: l.Runtime, Layout: host}
	if l.Runtime {
		out.RuntimeOffset, out.RuntimeStride, err = runtimeTail(resource.Type)
		out.MinimumByteSize = out.RuntimeOffset + out.RuntimeStride
	} else {
		out.ByteSize = roundUp(16, l.Size)
		out.MinimumByteSize = out.ByteSize
	}
	return out, err
}

func describeHostLayout(t *foundation.Type) (*HostLayout, error) {
	if t.Kind == foundation.AtomicKind {
		return describeHostLayout(t.Elem)
	}
	l, err := foundation.LayoutOf(t)
	if err != nil {
		return nil, err
	}
	out := &HostLayout{Size: l.Size, Stride: l.Stride, Runtime: l.Runtime}
	switch t.Kind {
	case foundation.BoolKind:
		out.Kind, out.Size = "bool", 4
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
		out.Elem, err = describeHostLayout(t.Elem)
	case foundation.FixedArrayKind:
		out.Kind, out.Count = "array", t.Count
		out.Elem, err = describeHostLayout(t.Elem)
	case foundation.RuntimeArrayKind:
		out.Kind = "runtime"
		out.Elem, err = describeHostLayout(t.Elem)
	case foundation.StructKind:
		out.Kind, out.Fields = "struct", []HostLayoutField{}
		for i, field := range t.Fields {
			child, childErr := describeHostLayout(field.Type)
			if childErr != nil {
				return nil, childErr
			}
			out.Fields = append(out.Fields, HostLayoutField{Name: field.Name, Offset: l.Fields[i].Offset, Type: child})
		}
	default:
		return nil, fmt.Errorf("cannot describe host type %s", t)
	}
	return out, err
}

func runtimeTail(t *foundation.Type) (uint32, uint32, error) {
	l, err := foundation.LayoutOf(t)
	if err != nil {
		return 0, 0, err
	}
	if t.Kind == foundation.RuntimeArrayKind {
		return 0, l.Stride, nil
	}
	if t.Kind != foundation.StructKind || len(t.Fields) == 0 {
		return 0, 0, fmt.Errorf("runtime type %s has no tail", t)
	}
	last := len(t.Fields) - 1
	return l.Fields[last].Offset, l.Fields[last].Layout.Stride, nil
}

func roundUp(alignment, size uint32) uint32 {
	if alignment == 0 {
		return size
	}
	return (size + alignment - 1) / alignment * alignment
}

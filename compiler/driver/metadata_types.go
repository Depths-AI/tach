package driver

import "encoding/json"

type Metadata struct {
	Schema   int                 `json:"schema"`
	Types    []TypeMetadata      `json:"types"`
	Programs []PublicProgramMeta `json:"programs"`
	Targets  TargetMetadata      `json:"targets"`
}
type TypeMetadata struct {
	Name   string      `json:"name"`
	Fields []FieldMeta `json:"fields"`
}
type FieldMeta struct {
	Name string `json:"name"`
	Type string `json:"type"`
}
type PublicProgramMeta struct {
	Name       string                 `json:"name"`
	Parameters []PublicParameterMeta  `json:"parameters"`
	Resources  []ExternalResourceMeta `json:"resources"`
	Launch     *LaunchMeta            `json:"launch,omitempty"`
	View       bool                   `json:"view,omitempty"`
}
type PublicParameterMeta struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"`
	Type     string `json:"type"`
	Resource *int   `json:"resource,omitempty"`
}
type LaunchMeta struct {
	Dimensions        int  `json:"dimensions"`
	InferFromResource *int `json:"inferFromResource,omitempty"`
}
type ExternalResourceMeta struct {
	Name            string      `json:"name"`
	Type            string      `json:"type"`
	ByteSize        uint32      `json:"byteSize,omitempty"`
	Alignment       uint32      `json:"alignment"`
	Runtime         bool        `json:"runtime"`
	RuntimeOffset   uint32      `json:"runtimeOffset,omitempty"`
	RuntimeStride   uint32      `json:"runtimeStride,omitempty"`
	MinimumByteSize uint32      `json:"minimumByteSize"`
	Layout          *HostLayout `json:"layout"`
}
type TargetMetadata struct {
	Web   json.RawMessage `json:"web"`
	SPIRV json.RawMessage `json:"spirv"`
}

type HostLayout struct {
	Kind    string            `json:"kind"`
	Size    uint32            `json:"size,omitempty"`
	Stride  uint32            `json:"stride,omitempty"`
	Count   uint32            `json:"count,omitempty"`
	Runtime bool              `json:"runtime,omitempty"`
	Elem    *HostLayout       `json:"elem,omitempty"`
	Fields  []HostLayoutField `json:"fields,omitempty"`
}
type HostLayoutField struct {
	Name   string      `json:"name"`
	Offset uint32      `json:"offset"`
	Type   *HostLayout `json:"type"`
}

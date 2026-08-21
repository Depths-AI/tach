package flow

import (
	"fmt"
	"slices"
	"strings"

	"tach/src/ir"
	"tach/src/source"
	"tach/src/types"
)

type ResourceID uint32
type VersionID uint32
type ShapeID uint32
type DispatchID uint32

type Module struct {
	Kernel        *ir.Module
	Programs      []*Program
	Documentation Documentation
}

type Documentation struct {
	Title     string
	Summary   string
	Types     map[string]TypeDocumentation
	Functions map[string]FunctionDocumentation
}
type TypeDocumentation struct {
	Summary string
	Fields  map[string]string
}
type FunctionDocumentation struct {
	Summary     string
	Parameters  map[string]string
	Coordinates map[string]string
	Returns     string
}

type ResourceAccess uint8

const (
	ReadAccess ResourceAccess = iota + 1
	WriteAccess
	ReadWriteAccess
	AtomicAccess
)

func (a ResourceAccess) String() string {
	return [...]string{"", "read", "write", "readWrite", "atomic"}[a]
}

func (m *Module) ProgramAccess(program *Program) map[ResourceID]ResourceAccess {
	type mode struct{ read, write, atomic bool }
	modes := map[ResourceID]mode{}
	for _, dispatch := range program.Dispatches {
		stage := m.Kernel.Function(dispatch.Stage)
		if stage == nil {
			continue
		}
		summary := ir.AnalyzeAccess(stage)
		for _, argument := range dispatch.Buffers {
			current, buffer := modes[argument.Resource], summary.Buffers[argument.Formal]
			current.read, current.write, current.atomic = current.read || buffer.Read, current.write || buffer.Write, current.atomic || buffer.Atomic
			modes[argument.Resource] = current
		}
	}
	out := map[ResourceID]ResourceAccess{}
	for resource, mode := range modes {
		switch {
		case mode.atomic:
			out[resource] = AtomicAccess
		case mode.read && mode.write:
			out[resource] = ReadWriteAccess
		case mode.write:
			out[resource] = WriteAccess
		default:
			out[resource] = ReadAccess
		}
	}
	return out
}

type ParameterKind uint8

const (
	BufferParameter ParameterKind = iota + 1
	ValueParameter
)

type Parameter struct {
	Name     string
	Kind     ParameterKind
	Type     *types.Type
	Resource ResourceID
	Span     source.Span
}

type ResourceKind uint8

const (
	External ResourceKind = iota + 1
	Transient
)

type Resource struct {
	ID        ResourceID
	Name      string
	Kind      ResourceKind
	Type      *types.Type
	Parameter int
	Length    ShapeID
	Initial   VersionID
	Final     VersionID
	Span      source.Span
}

type Version struct {
	ID       VersionID
	Resource ResourceID
	Previous VersionID
	Producer DispatchID
	Defined  bool
}

type ShapeOp uint8

const (
	ShapeConstant ShapeOp = iota + 1
	ShapeParameter
	ShapeResourceLength
	ShapeLaunchAxis
	ShapeAdd
	ShapeSub
	ShapeMul
	ShapeDiv
	ShapeRem
	ShapeMin
	ShapeMax
	ShapeCeilDiv
)

type Shape struct {
	ID        ShapeID
	Op        ShapeOp
	Value     uint32
	Parameter int
	Resource  ResourceID
	Path      []string
	Axis      uint8
	Left      ShapeID
	Right     ShapeID
	Span      source.Span
}

type BufferArgument struct {
	Formal   int
	Resource ResourceID
	Input    VersionID
	Output   VersionID
}

type ValueKind uint8

const (
	ValueParameterRef ValueKind = iota + 1
	ValueConstant
	ValueShape
	ValueRepeat
)

type ValueArgument struct {
	Formal    int
	Kind      ValueKind
	Parameter int
	Path      []string
	Constant  *types.Value
	Shape     ShapeID
}

type Dispatch struct {
	ID      DispatchID
	Stage   string
	Domain  []ShapeID
	Buffers []BufferArgument
	Values  []ValueArgument
	Span    source.Span
}

type ViewFormat uint8

const (
	SRGB8 ViewFormat = iota + 1
)

func (f ViewFormat) String() string {
	if f == SRGB8 {
		return "srgb8"
	}
	return fmt.Sprintf("viewFormat(%d)", f)
}

type View struct {
	Format ViewFormat
	Source ResourceID
	Input  VersionID
	Width  ShapeID
	Height ShapeID
	Span   source.Span
}

type Program struct {
	Name       string
	Span       source.Span
	Indexed    bool
	Rank       int
	Parameters []Parameter
	Resources  []Resource
	Versions   []Version
	Shapes     []Shape
	Dispatches []Dispatch
	View       *View
	nextRes    ResourceID
	nextVer    VersionID
	nextShape  ShapeID
	nextDisp   DispatchID
}

func (p *Program) AddResource(r Resource) ResourceID {
	p.nextRes++
	r.ID = p.nextRes
	p.Resources = append(p.Resources, r)
	return r.ID
}

func (p *Program) AddVersion(v Version) VersionID {
	p.nextVer++
	v.ID = p.nextVer
	p.Versions = append(p.Versions, v)
	return v.ID
}

func (p *Program) AddShape(s Shape) ShapeID {
	for _, existing := range p.Shapes {
		if existing.Op == s.Op && existing.Value == s.Value && existing.Parameter == s.Parameter && existing.Resource == s.Resource && existing.Axis == s.Axis && existing.Left == s.Left && existing.Right == s.Right && slices.Equal(existing.Path, s.Path) {
			return existing.ID
		}
	}
	p.nextShape++
	s.ID = p.nextShape
	p.Shapes = append(p.Shapes, s)
	return s.ID
}

func (p *Program) AddDispatch(d Dispatch) DispatchID {
	p.nextDisp++
	d.ID = p.nextDisp
	p.Dispatches = append(p.Dispatches, d)
	return d.ID
}

func (p *Program) Resource(id ResourceID) *Resource {
	if id == 0 || int(id) > len(p.Resources) || p.Resources[id-1].ID != id {
		return nil
	}
	return &p.Resources[id-1]
}

func (p *Program) Version(id VersionID) *Version {
	if id == 0 || int(id) > len(p.Versions) || p.Versions[id-1].ID != id {
		return nil
	}
	return &p.Versions[id-1]
}

func (p *Program) Shape(id ShapeID) *Shape {
	if id == 0 || int(id) > len(p.Shapes) || p.Shapes[id-1].ID != id {
		return nil
	}
	return &p.Shapes[id-1]
}

func Clone(m *Module) *Module {
	if m == nil {
		return nil
	}
	out := &Module{Kernel: ir.Clone(m.Kernel), Programs: make([]*Program, len(m.Programs))}
	for i, p := range m.Programs {
		q := *p
		q.Parameters = append([]Parameter(nil), p.Parameters...)
		q.Resources = append([]Resource(nil), p.Resources...)
		q.Versions = append([]Version(nil), p.Versions...)
		q.Shapes = append([]Shape(nil), p.Shapes...)
		for j := range q.Shapes {
			q.Shapes[j].Path = append([]string(nil), p.Shapes[j].Path...)
		}
		q.Dispatches = append([]Dispatch(nil), p.Dispatches...)
		for j := range q.Dispatches {
			q.Dispatches[j].Domain = append([]ShapeID(nil), p.Dispatches[j].Domain...)
			q.Dispatches[j].Buffers = append([]BufferArgument(nil), p.Dispatches[j].Buffers...)
			q.Dispatches[j].Values = append([]ValueArgument(nil), p.Dispatches[j].Values...)
			for k := range q.Dispatches[j].Values {
				value := &q.Dispatches[j].Values[k]
				if value.Constant != nil {
					value.Constant = &types.Value{Type: value.Constant.Type, Bits: append([]uint32(nil), value.Constant.Bits...)}
				}
				value.Path = append([]string(nil), value.Path...)
			}
		}
		if p.View != nil {
			view := *p.View
			q.View = &view
		}
		out.Programs[i] = &q
	}
	return out
}

func Dump(m *Module) string {
	var b strings.Builder
	for _, p := range m.Programs {
		fmt.Fprintf(&b, "program @%s(", p.Name)
		for i, parameter := range p.Parameters {
			if i > 0 {
				b.WriteString(", ")
			}
			if parameter.Kind == BufferParameter {
				fmt.Fprintf(&b, "%s=%%r%d: buffer<%s>", parameter.Name, parameter.Resource, parameter.Type)
			} else {
				fmt.Fprintf(&b, "%s: %s", parameter.Name, parameter.Type)
			}
		}
		b.WriteString(") {\n")
		for _, resource := range p.Resources {
			fmt.Fprintf(&b, "  resource %%r%d %s kind=%s initial=%%v%d final=%%v%d", resource.ID, resource.Name, resourceKind(resource.Kind), resource.Initial, resource.Final)
			if resource.Kind == Transient {
				fmt.Fprintf(&b, " length=%%s%d", resource.Length)
			}
			b.WriteByte('\n')
		}
		for _, version := range p.Versions {
			fmt.Fprintf(&b, "  version %%v%d resource=%%r%d previous=%%v%d producer=%%d%d defined=%t\n", version.ID, version.Resource, version.Previous, version.Producer, version.Defined)
		}
		for _, shape := range p.Shapes {
			fmt.Fprintf(&b, "  shape %%s%d = %s\n", shape.ID, dumpShape(p, shape.ID, map[ShapeID]bool{}))
		}
		for _, dispatch := range p.Dispatches {
			fmt.Fprintf(&b, "  dispatch %%d%d @%s over [", dispatch.ID, dispatch.Stage)
			for i, axis := range dispatch.Domain {
				if i > 0 {
					b.WriteString(", ")
				}
				fmt.Fprintf(&b, "%%s%d", axis)
			}
			b.WriteString("]\n")
		}
		if p.View != nil {
			fmt.Fprintf(&b, "  view %s %%r%d version=%%v%d extent=[%%s%d, %%s%d]\n", p.View.Format, p.View.Source, p.View.Input, p.View.Width, p.View.Height)
		}
		b.WriteString("}\n\n")
	}
	return b.String()
}

func resourceKind(kind ResourceKind) string {
	if kind == External {
		return "external"
	}
	return "transient"
}

func dumpShape(p *Program, id ShapeID, active map[ShapeID]bool) string {
	if active[id] {
		return "<cycle>"
	}
	s := p.Shape(id)
	if s == nil {
		return "<invalid>"
	}
	active[id] = true
	defer delete(active, id)
	switch s.Op {
	case ShapeConstant:
		return fmt.Sprintf("%d", s.Value)
	case ShapeParameter:
		return fmt.Sprintf("parameter(%d%s)", s.Parameter, dumpPath(s.Path))
	case ShapeResourceLength:
		return fmt.Sprintf("length(%%r%d%s)", s.Resource, dumpPath(s.Path))
	case ShapeLaunchAxis:
		return fmt.Sprintf("launch(%d)", s.Axis)
	default:
		return fmt.Sprintf("%s(%s, %s)", s.Op, dumpShape(p, s.Left, active), dumpShape(p, s.Right, active))
	}
}

func dumpPath(path []string) string {
	if len(path) == 0 {
		return ""
	}
	return "." + strings.Join(path, ".")
}

func (op ShapeOp) String() string {
	switch op {
	case ShapeAdd:
		return "add"
	case ShapeSub:
		return "sub"
	case ShapeMul:
		return "mul"
	case ShapeDiv:
		return "div"
	case ShapeRem:
		return "rem"
	case ShapeMin:
		return "min"
	case ShapeMax:
		return "max"
	case ShapeCeilDiv:
		return "ceilDiv"
	}
	return fmt.Sprintf("shape(%d)", op)
}

package spirv

import (
	"encoding/binary"
	"fmt"
	"tach/foundation"
	"tach/ir"
)

type inputKind uint8

const (
	inputGlobalIndex inputKind = iota + 1
	inputLocalIndex
	inputLocalLinear
)

type program struct {
	executable *plan
	source     *ir.KernelModule
	functions  map[*ir.Function]*ir.Coordinates
	kernels    map[*ir.Function]*physicalKernel
}

type Result struct {
	Binary       []byte
	KernelModule *ir.KernelModule
	RuntimeJSON  []byte
}

func Lower(logical *ir.Module) (*Result, error) {
	executable, err := planModule(logical)
	if err != nil {
		return nil, err
	}
	binary, err := emitModule(executable)
	if err != nil {
		return nil, err
	}
	runtimeJSON, err := encodeRuntime(executable)
	if err != nil {
		return nil, err
	}
	return &Result{Binary: binary, KernelModule: executable.KernelModule, RuntimeJSON: runtimeJSON}, nil
}

func lower(executable *plan) (*program, error) {
	functions, kernels, err := executable.IndexFunctions()
	if err != nil {
		return nil, err
	}
	return &program{executable: executable, source: executable.KernelModule, functions: functions, kernels: kernels}, nil
}

func inputs(_ *ir.Function, coordinates *ir.Coordinates) map[inputKind]bool {
	used := map[inputKind]bool{}
	for id, coordinate := range coordinates.Values {
		if coordinates.Uses[id] == 0 {
			continue
		}
		switch coordinate.Space {
		case ir.Global:
			used[inputGlobalIndex] = true
		case ir.Local:
			used[inputLocalIndex] = true
		case ir.LocalLinear:
			used[inputLocalLinear] = true
		}
	}
	return used
}

func coordinate(f *ir.Coordinates, id ir.ValueID) (inputKind, uint32) {
	coordinate := f.Values[id]
	switch coordinate.Space {
	case ir.Global:
		return inputGlobalIndex, uint32(coordinate.Dimension)
	case ir.Local:
		return inputLocalIndex, uint32(coordinate.Dimension)
	case ir.LocalLinear:
		return inputLocalLinear, 0
	}
	panic("unknown lowered coordinate space")
}

// Emit lowers verified Tach IR to a SPIR-V 1.6 compute module and immediately
// parses and validates the produced binary with Tach's own SPIR-V validator.
func emitModule(executable *plan) ([]byte, error) {
	if err := verify(executable); err != nil {
		return nil, fmt.Errorf("executable verification: %w", err)
	}
	p, err := lower(executable)
	if err != nil {
		return nil, err
	}
	b := newBuilder(p)
	if err := b.build(); err != nil {
		return nil, err
	}
	words := b.words()
	out := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(out[i*4:], w)
	}
	if err := Validate(out); err != nil {
		return nil, fmt.Errorf("tach SPIR-V self-validation failed: %w", err)
	}
	return out, nil
}

type builder struct {
	p *program
	m *ir.KernelModule

	nextID uint32

	capabilities []uint32
	extImports   []uint32
	memoryModel  []uint32
	entryPoints  []uint32
	execModes    []uint32
	debug        []uint32
	annotations  []uint32
	typesGlobals []uint32
	functions    []uint32

	types           map[string]uint32
	pointers        map[string]uint32
	fnTypes         map[string]uint32
	constants       map[string]uint32
	funcIDs         map[string]uint32
	resourceIDs     map[*ir.Function][]uint32
	parameterBlocks map[*ir.Function]*ir.HostParameterBlock
	parameterIDs    map[*ir.Function]uint32
	inputIDs        map[inputKind]uint32
	workgroupIDs    map[string][]uint32
	globalUses      map[string]map[uint32]bool
	calls           map[string]map[string]bool
	glsl450         uint32
}

type typeRole uint8

const (
	typeLogical typeRole = iota
	typeHostABI
)

func newBuilder(p *program) *builder {
	return &builder{
		p:               p,
		m:               p.source,
		nextID:          1,
		types:           map[string]uint32{},
		pointers:        map[string]uint32{},
		fnTypes:         map[string]uint32{},
		constants:       map[string]uint32{},
		funcIDs:         map[string]uint32{},
		resourceIDs:     map[*ir.Function][]uint32{},
		parameterBlocks: map[*ir.Function]*ir.HostParameterBlock{},
		parameterIDs:    map[*ir.Function]uint32{},
		inputIDs:        map[inputKind]uint32{},
		workgroupIDs:    map[string][]uint32{},
		globalUses:      map[string]map[uint32]bool{},
		calls:           map[string]map[string]bool{},
	}
}

func (b *builder) id() uint32 {
	id := b.nextID
	b.nextID++
	return id
}

func emit(dst *[]uint32, op Op, operands ...uint32) int {
	start := len(*dst)
	wc := uint32(len(operands) + 1)
	*dst = append(*dst, wc<<16|uint32(op))
	*dst = append(*dst, operands...)
	return start
}

func encodeString(s string) []uint32 {
	buf := append([]byte(s), 0)
	for len(buf)%4 != 0 {
		buf = append(buf, 0)
	}
	words := make([]uint32, len(buf)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(buf[i*4:])
	}
	return words
}

func (b *builder) build() error {
	emit(&b.capabilities, OpCapability, CapabilityShader)
	if ir.UsesKind(b.m, foundation.Float16Kind) {
		emit(&b.capabilities, OpCapability, CapabilityFloat16)
		features := requiredFeatures(b.p.executable)
		for _, feature := range features {
			switch feature {
			case storageBuffer16BitAccess:
				emit(&b.capabilities, OpCapability, CapabilityStorageBuffer16BitAccess)
			case uniformAndStorage16BitAccess:
				emit(&b.capabilities, OpCapability, CapabilityUniformAndStorage16BitAccess)
			}
		}
	}
	emit(&b.capabilities, OpCapability, CapabilityVulkanMemoryModel)
	emit(&b.memoryModel, OpMemoryModel, AddressingLogical, MemoryVulkan)

	// Function IDs must exist before entry-point declarations and forward calls.
	for _, f := range b.m.Functions {
		b.funcIDs[f.Name] = b.id()
		emit(&b.debug, OpName, append([]uint32{b.funcIDs[f.Name]}, encodeString(f.Name)...)...)
	}

	if err := b.emitStructDebugTypes(); err != nil {
		return err
	}
	if err := b.emitResources(); err != nil {
		return err
	}
	if err := b.emitParameterBlocks(); err != nil {
		return err
	}
	if err := b.emitInputs(); err != nil {
		return err
	}
	if err := b.emitWorkgroups(); err != nil {
		return err
	}
	for _, f := range b.m.Functions {
		if err := b.emitFunction(f); err != nil {
			return fmt.Errorf("SPIR-V lower %s: %w", f.Name, err)
		}
	}
	return b.emitEntryPoints()
}

func (b *builder) words() []uint32 {
	out := []uint32{Magic, Version, 0, b.nextID, 0}
	for _, s := range [][]uint32{b.capabilities, b.extImports, b.memoryModel, b.entryPoints, b.execModes, b.debug, b.annotations, b.typesGlobals, b.functions} {
		out = append(out, s...)
	}
	return out
}

func (b *builder) ensureGLSL450() uint32 {
	if b.glsl450 != 0 {
		return b.glsl450
	}
	b.glsl450 = b.id()
	ops := append([]uint32{b.glsl450}, encodeString("GLSL.std.450")...)
	emit(&b.extImports, OpExtInstImport, ops...)
	return b.glsl450
}

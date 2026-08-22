package spirv

import (
	"fmt"
	"sort"
	"strconv"
	"tach/foundation"
	"tach/ir"
)

func (b *builder) emitStructDebugTypes() error {
	for _, t := range b.m.Structs {
		if foundation.ContainsRuntimeArray(t) {
			continue
		}
		id, err := b.typeID(t, typeLogical)
		if err != nil {
			return err
		}
		emit(&b.debug, OpName, append([]uint32{id}, encodeString(t.Name)...)...)
		for i, f := range t.Fields {
			ops := []uint32{id, uint32(i)}
			ops = append(ops, encodeString(f.Name)...)
			emit(&b.debug, OpMemberName, ops...)
		}
	}
	return nil
}

func (b *builder) emitResources() error {
	for kernelIndex := range b.p.executable.PhysicalKernels {
		kernel := &b.p.executable.PhysicalKernels[kernelIndex]
		b.resourceIDs[kernel.Function] = make([]uint32, len(kernel.Function.BufferParams))
		for i, r := range kernel.Function.BufferParams {
			physical, err := b.typeID(r.Type, typeHostABI)
			if err != nil {
				return fmt.Errorf("resource %s type: %w", r.Name, err)
			}
			root := physical
			if r.Type.Kind == foundation.StructKind && foundation.ContainsRuntimeArray(r.Type) {
				emit(&b.annotations, OpDecorate, root, DecorationBlock)
			} else {
				root = b.id()
				emit(&b.typesGlobals, OpTypeStruct, root, physical)
				emit(&b.annotations, OpDecorate, root, DecorationBlock)
				emit(&b.annotations, OpMemberDecorate, root, 0, DecorationOffset, 0)
				emit(&b.debug, OpName, append([]uint32{root}, encodeString(fmt.Sprintf("__tach_resource_%d_%d", kernelIndex, i))...)...)
				emit(&b.debug, OpMemberName, append([]uint32{root, 0}, encodeString("data")...)...)
			}

			storage := uint32(StorageStorageBuffer)
			ptr := b.id()
			emit(&b.typesGlobals, OpTypePointer, ptr, storage, root)
			varID := b.id()
			b.resourceIDs[kernel.Function][i] = varID
			emit(&b.typesGlobals, OpVariable, ptr, varID, storage)
			emit(&b.annotations, OpDecorate, varID, DecorationDescriptorSet, 0)
			emit(&b.annotations, OpDecorate, varID, DecorationBinding, uint32(i))
			if r.Access == ir.Read && !foundation.ContainsAtomic(r.Type) {
				emit(&b.annotations, OpDecorate, varID, DecorationNonWritable)
			}
			emit(&b.debug, OpName, append([]uint32{varID}, encodeString(r.Name)...)...)
		}
	}
	return nil
}

func (b *builder) emitParameterBlocks() error {
	for i := range b.p.executable.PhysicalKernels {
		block := b.p.executable.PhysicalKernels[i].Parameters
		if block == nil {
			continue
		}
		b.parameterBlocks[block.Function] = block
		typeID, err := b.typeID(block.Type, typeHostABI)
		if err != nil {
			return fmt.Errorf("kernel %s parameter block type: %w", block.Function.Name, err)
		}
		emit(&b.annotations, OpDecorate, typeID, DecorationBlock)
		emit(&b.debug, OpName, append([]uint32{typeID}, encodeString(block.Type.Name)...)...)
		for index, field := range block.Fields {
			emit(&b.debug, OpMemberName, append([]uint32{typeID, uint32(index)}, encodeString(field.Name)...)...)
		}
		pointer, err := b.pointerID(StorageUniform, block.Type)
		if err != nil {
			return err
		}
		variable := b.id()
		b.parameterIDs[block.Function] = variable
		emit(&b.typesGlobals, OpVariable, pointer, variable, StorageUniform)
		emit(&b.annotations, OpDecorate, variable, DecorationDescriptorSet, 0)
		emit(&b.annotations, OpDecorate, variable, DecorationBinding, block.Binding)
		emit(&b.debug, OpName, append([]uint32{variable}, encodeString("__tach_parameters_"+block.Function.Name)...)...)
	}
	return nil
}

func inputInfo(k inputKind) (*foundation.Type, uint32, string) {
	vec3u := foundation.VectorOf(foundation.Uint32Type, 3)
	switch k {
	case inputGlobalIndex:
		return vec3u, BuiltInGlobalInvocationID, "globalIndex"
	case inputLocalIndex:
		return vec3u, BuiltInLocalInvocationID, "localIndex"
	case inputLocalLinear:
		return foundation.Uint32Type, BuiltInLocalInvocationIndex, "localLinear"
	default:
		return nil, 0, ""
	}
}

func (b *builder) emitInputs() error {
	used := map[inputKind]bool{}
	for _, f := range b.m.Functions {
		for k := range inputs(f, b.p.functions[f]) {
			used[k] = true
		}
	}
	order := []inputKind{inputGlobalIndex, inputLocalIndex, inputLocalLinear}
	for _, k := range order {
		if !used[k] {
			continue
		}
		t, decoration, name := inputInfo(k)
		ptr, err := b.pointerID(StorageInput, t)
		if err != nil {
			return err
		}
		id := b.id()
		b.inputIDs[k] = id
		emit(&b.typesGlobals, OpVariable, ptr, id, StorageInput)
		emit(&b.annotations, OpDecorate, id, DecorationBuiltIn, decoration)
		emit(&b.debug, OpName, append([]uint32{id}, encodeString("__tach_"+name)...)...)
	}
	return nil
}

func (b *builder) emitWorkgroups() error {
	for _, f := range b.m.Functions {
		if f.Kind != ir.Stage || len(f.WorkgroupVars) == 0 {
			continue
		}
		ids := make([]uint32, len(f.WorkgroupVars))
		for i, w := range f.WorkgroupVars {
			ptr, err := b.pointerID(StorageWorkgroup, w.Type)
			if err != nil {
				return fmt.Errorf("workgroup %s.%s type: %w", f.Name, w.Name, err)
			}
			id := b.id()
			ids[i] = id
			zero, err := b.nullConstant(w.Type)
			if err != nil {
				return fmt.Errorf("workgroup %s.%s initializer: %w", f.Name, w.Name, err)
			}
			emit(&b.typesGlobals, OpVariable, ptr, id, StorageWorkgroup, zero)
			emit(&b.debug, OpName, append([]uint32{id}, encodeString("__tach_w_"+f.Name+"_"+strconv.Itoa(i))...)...)
		}
		b.workgroupIDs[f.Name] = ids
	}
	return nil
}

func (b *builder) emitEntryPoints() error {
	for _, f := range b.m.Functions {
		if f.Kind != ir.Stage {
			continue
		}
		ops := []uint32{ExecutionModelGLCompute, b.funcIDs[f.Name]}
		ops = append(ops, encodeString(f.Name)...)
		globals, err := b.entryGlobals(f.Name, map[string]bool{}, map[string]bool{})
		if err != nil {
			return err
		}
		ops = append(ops, globals...)
		emit(&b.entryPoints, OpEntryPoint, ops...)
		workgroup := b.p.kernels[f].Workgroup
		emit(&b.execModes, OpExecutionMode, b.funcIDs[f.Name], ExecutionModeLocalSize, workgroup[0], workgroup[1], workgroup[2])
	}
	return nil
}

func (b *builder) entryGlobals(name string, visiting, visited map[string]bool) ([]uint32, error) {
	used := map[uint32]bool{}
	var walk func(string) error
	walk = func(function string) error {
		if visiting[function] {
			return fmt.Errorf("recursive static call graph at %s", function)
		}
		if visited[function] {
			return nil
		}
		if b.funcIDs[function] == 0 {
			return fmt.Errorf("static call graph references unknown function %s", function)
		}
		visiting[function] = true
		for id := range b.globalUses[function] {
			used[id] = true
		}
		for callee := range b.calls[function] {
			if err := walk(callee); err != nil {
				return err
			}
		}
		delete(visiting, function)
		visited[function] = true
		return nil
	}
	if err := walk(name); err != nil {
		return nil, err
	}
	ids := make([]uint32, 0, len(used))
	for id := range used {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids, nil
}

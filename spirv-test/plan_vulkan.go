package spirvtest

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"unsafe"

	vk "github.com/goki/vulkan"
)

type programArguments struct {
	Buffers map[string][]byte
	Values  map[string]any
	Launch  [3]uint32
	Repeat  uint32
}

type allocatedBuffer struct {
	name     string
	data     []byte
	readback bool
	buffer   vk.Buffer
	memory   vk.DeviceMemory
	size     vk.DeviceSize
	coherent bool
}

type physicalPipeline struct {
	metadata kernelMetadata
	set      vk.DescriptorSetLayout
	layout   vk.PipelineLayout
	pipeline vk.Pipeline
}

type preparedDispatch struct {
	step     stepMetadata
	pipeline *physicalPipeline
	set      vk.DescriptorSet
}

func (h *vulkanHarness) executeProgram(spirv, metadataJSON []byte, name string, arguments programArguments) (output map[string][]byte, err error) {
	mark := h.validationMark()
	output, err = h.executeProgramUnchecked(spirv, metadataJSON, name, arguments)
	return output, errors.Join(err, validationError(h.validationSince(mark)))
}

func (h *vulkanHarness) executeProgramUnchecked(spirv, metadataJSON []byte, name string, arguments programArguments) (map[string][]byte, error) {
	if len(spirv) < 20 || len(spirv)%4 != 0 {
		return nil, fmt.Errorf("invalid SPIR-V byte length %d", len(spirv))
	}
	var metadata moduleMetadata
	if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
		return nil, fmt.Errorf("decode Tach metadata: %w", err)
	}
	if metadata.Schema != 1 || metadata.Targets.SPIRV == nil {
		return nil, errors.New("metadata has no schema-1 SPIR-V target")
	}
	programIndex := -1
	for index := range metadata.Programs {
		if metadata.Programs[index].Name == name {
			programIndex = index
			break
		}
	}
	if programIndex < 0 {
		return nil, fmt.Errorf("program %q is absent from metadata", name)
	}
	program := &metadata.Programs[programIndex]
	plan := &metadata.Targets.SPIRV.Programs[programIndex]
	if plan.Program != programIndex {
		return nil, fmt.Errorf("program %q plan index mismatch", name)
	}
	if arguments.Repeat == 0 {
		arguments.Repeat = 1
	}

	values := make([]any, len(program.Parameters))
	external := make([]*allocatedBuffer, len(program.Resources))
	allocations := []*allocatedBuffer{}
	destroyAllocations := func() {
		for _, allocation := range allocations {
			vk.DestroyBuffer(h.device, allocation.buffer, nil)
			vk.FreeMemory(h.device, allocation.memory, nil)
		}
	}
	defer destroyAllocations()
	for parameterIndex, parameter := range program.Parameters {
		if parameter.Kind == "value" {
			value, ok := arguments.Values[parameter.Name]
			if !ok {
				return nil, fmt.Errorf("program value %s has no test data", parameter.Name)
			}
			values[parameterIndex] = value
			continue
		}
		if parameter.Kind != "buffer" || parameter.Resource == nil || *parameter.Resource < 0 || *parameter.Resource >= len(program.Resources) {
			return nil, fmt.Errorf("program buffer %s has invalid metadata", parameter.Name)
		}
		resource := program.Resources[*parameter.Resource]
		data, ok := arguments.Buffers[parameter.Name]
		if !ok {
			return nil, fmt.Errorf("program buffer %s has no test data", parameter.Name)
		}
		if len(data) < int(resource.MinimumByteSize) || resource.ByteSize != 0 && len(data) != int(resource.ByteSize) {
			return nil, fmt.Errorf("program buffer %s has invalid byte length %d", parameter.Name, len(data))
		}
		allocation, err := h.allocate(parameter.Name, "storage", data, true)
		if err != nil {
			return nil, err
		}
		external[*parameter.Resource] = allocation
		allocations = append(allocations, allocation)
	}
	if len(arguments.Buffers) != len(external) {
		return nil, fmt.Errorf("program %s received %d buffers, want %d", name, len(arguments.Buffers), len(external))
	}

	transientBytes := make([]uint32, len(plan.Transients))
	colors := map[int]*allocatedBuffer{}
	colorBytes := map[int]uint32{}
	for index, transient := range plan.Transients {
		length, err := evaluateShape(transient.Length, program, values, arguments.Buffers, arguments.Launch)
		if err != nil || length == 0 || uint64(length)*uint64(transient.Stride) > math.MaxUint32 {
			return nil, fmt.Errorf("transient %d has invalid length: %w", index, err)
		}
		transientBytes[index] = max(transient.MinimumByteSize, length*transient.Stride)
		colorBytes[transient.Color] = max(colorBytes[transient.Color], transientBytes[index])
	}
	for color, byteSize := range colorBytes {
		allocation, err := h.allocate(fmt.Sprintf("scratch %d", color), "storage", make([]byte, byteSize), false)
		if err != nil {
			return nil, err
		}
		colors[color] = allocation
		allocations = append(allocations, allocation)
	}

	words := make([]uint32, len(spirv)/4)
	for index := range words {
		words[index] = binary.LittleEndian.Uint32(spirv[index*4:])
	}
	shaderInfo := vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint64(len(spirv)), PCode: words}
	var shader vk.ShaderModule
	if err := result("create SPIR-V shader module", vk.CreateShaderModule(h.device, &shaderInfo, nil, &shader)); err != nil {
		return nil, err
	}
	defer vk.DestroyShaderModule(h.device, shader, nil)

	pipelines := map[int]*physicalPipeline{}
	defer func() {
		for _, pipeline := range pipelines {
			vk.DestroyPipeline(h.device, pipeline.pipeline, nil)
			vk.DestroyPipelineLayout(h.device, pipeline.layout, nil)
			vk.DestroyDescriptorSetLayout(h.device, pipeline.set, nil)
		}
	}()
	for _, step := range plan.Steps {
		if step.Kind != "dispatch" {
			continue
		}
		if pipelines[step.Kernel] != nil || step.Kernel < 0 || step.Kernel >= len(metadata.Targets.SPIRV.Kernels) {
			continue
		}
		kernel := metadata.Targets.SPIRV.Kernels[step.Kernel]
		bindings := make([]vk.DescriptorSetLayoutBinding, 0, len(kernel.Bindings)+1)
		for _, binding := range kernel.Bindings {
			if binding.Group != 0 {
				return nil, errors.New("Vulkan harness supports target set zero")
			}
			bindings = append(bindings, descriptorLayoutBinding(binding.Binding, vk.DescriptorTypeStorageBuffer))
		}
		if kernel.ParameterBlock != nil {
			bindings = append(bindings, descriptorLayoutBinding(kernel.ParameterBlock.Binding, vk.DescriptorTypeUniformBuffer))
		}
		sort.Slice(bindings, func(i, j int) bool { return bindings[i].Binding < bindings[j].Binding })
		setInfo := vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}
		pipeline := &physicalPipeline{metadata: kernel}
		var setLayout vk.DescriptorSetLayout
		if err := result("create descriptor set layout", vk.CreateDescriptorSetLayout(h.device, &setInfo, nil, &setLayout)); err != nil {
			return nil, err
		}
		pipeline.set = setLayout
		layoutInfo := vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{pipeline.set}}
		var pipelineLayout vk.PipelineLayout
		if err := result("create pipeline layout", vk.CreatePipelineLayout(h.device, &layoutInfo, nil, &pipelineLayout)); err != nil {
			return nil, err
		}
		pipeline.layout = pipelineLayout
		pipelineInfo := vk.ComputePipelineCreateInfo{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: kernel.EntryPoint + "\x00"}, Layout: pipeline.layout}
		var noCache vk.PipelineCache
		created := make([]vk.Pipeline, 1)
		if err := result("create compute pipeline", vk.CreateComputePipelines(h.device, noCache, 1, []vk.ComputePipelineCreateInfo{pipelineInfo}, nil, created)); err != nil {
			return nil, err
		}
		pipeline.pipeline = created[0]
		pipelines[step.Kernel] = pipeline
	}

	dispatchCount := 0
	storageDescriptors, uniformDescriptors := uint32(0), uint32(0)
	for _, step := range plan.Steps {
		if step.Kind == "dispatch" {
			dispatchCount++
			storageDescriptors += uint32(len(step.Resources))
			if pipelines[step.Kernel].metadata.ParameterBlock != nil {
				uniformDescriptors++
			}
		}
	}
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: storageDescriptors}}
	if uniformDescriptors > 0 {
		poolSizes = append(poolSizes, vk.DescriptorPoolSize{Type: vk.DescriptorTypeUniformBuffer, DescriptorCount: uniformDescriptors})
	}
	poolInfo := vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: uint32(dispatchCount), PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}
	var descriptorPool vk.DescriptorPool
	if err := result("create descriptor pool", vk.CreateDescriptorPool(h.device, &poolInfo, nil, &descriptorPool)); err != nil {
		return nil, err
	}
	defer vk.DestroyDescriptorPool(h.device, descriptorPool, nil)

	prepared := make([]preparedDispatch, len(plan.Steps))
	for stepIndex, step := range plan.Steps {
		if step.Kind != "dispatch" {
			continue
		}
		pipeline := pipelines[step.Kernel]
		allocateInfo := vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: descriptorPool, DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{pipeline.set}}
		sets := make([]vk.DescriptorSet, 1)
		if err := result("allocate descriptor set", vk.AllocateDescriptorSets(h.device, &allocateInfo, &sets[0])); err != nil {
			return nil, err
		}
		writes := make([]vk.WriteDescriptorSet, 0, len(step.Resources)+1)
		for _, source := range step.Resources {
			allocation, byteSize, err := resolveResource(source, external, plan.Transients, colors, transientBytes)
			if err != nil {
				return nil, err
			}
			writes = append(writes, descriptorWrite(sets[0], source.Binding, vk.DescriptorTypeStorageBuffer, allocation.buffer, byteSize))
		}
		if block := pipeline.metadata.ParameterBlock; block != nil {
			bytes, err := packParameterBlock(*block, step.Parameters, program, values, arguments.Buffers, arguments.Launch, arguments.Repeat)
			if err != nil {
				return nil, err
			}
			allocation, err := h.allocate(fmt.Sprintf("parameters %d", stepIndex), "uniform", bytes, false)
			if err != nil {
				return nil, err
			}
			allocations = append(allocations, allocation)
			writes = append(writes, descriptorWrite(sets[0], block.Binding, vk.DescriptorTypeUniformBuffer, allocation.buffer, block.ByteSize))
		}
		vk.UpdateDescriptorSets(h.device, uint32(len(writes)), writes, 0, nil)
		prepared[stepIndex] = preparedDispatch{step: step, pipeline: pipeline, set: sets[0]}
	}

	commandPoolInfo := vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, Flags: vk.CommandPoolCreateFlags(vk.CommandPoolCreateTransientBit), QueueFamilyIndex: h.queueFamily}
	var commandPool vk.CommandPool
	if err := result("create command pool", vk.CreateCommandPool(h.device, &commandPoolInfo, nil, &commandPool)); err != nil {
		return nil, err
	}
	defer vk.DestroyCommandPool(h.device, commandPool, nil)
	commandBuffers := make([]vk.CommandBuffer, 1)
	commandBufferInfo := vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: commandPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}
	if err := result("allocate command buffer", vk.AllocateCommandBuffers(h.device, &commandBufferInfo, commandBuffers)); err != nil {
		return nil, err
	}
	commandBuffer := commandBuffers[0]
	beginInfo := vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo, Flags: vk.CommandBufferUsageFlags(vk.CommandBufferUsageOneTimeSubmitBit)}
	if err := result("begin command buffer", vk.BeginCommandBuffer(commandBuffer, &beginInfo)); err != nil {
		return nil, err
	}
	repetitions := arguments.Repeat
	if plan.Repeat == "invocation-loop" {
		repetitions = 1
	}
	for repetition := uint32(0); repetition < repetitions; repetition++ {
		for stepIndex, step := range plan.Steps {
			if step.Kind == "barrier" {
				recordComputeBarrier(commandBuffer)
				continue
			}
			dispatch := prepared[stepIndex]
			vk.CmdBindPipeline(commandBuffer, vk.PipelineBindPointCompute, dispatch.pipeline.pipeline)
			vk.CmdBindDescriptorSets(commandBuffer, vk.PipelineBindPointCompute, dispatch.pipeline.layout, 0, 1, []vk.DescriptorSet{dispatch.set}, 0, nil)
			domain := [3]uint32{1, 1, 1}
			for axis, expression := range step.Domain {
				value, err := evaluateShape(expression, program, values, arguments.Buffers, arguments.Launch)
				if err != nil || value == 0 {
					return nil, fmt.Errorf("dispatch domain %d: %w", axis, err)
				}
				domain[axis] = value
			}
			workgroup := dispatch.pipeline.metadata.WorkgroupSize
			vk.CmdDispatch(commandBuffer, ceilDiv(domain[0], workgroup[0]), ceilDiv(domain[1], workgroup[1]), ceilDiv(domain[2], workgroup[2]))
		}
		if repetition+1 < repetitions && plan.RepeatBarrier != nil {
			recordComputeBarrier(commandBuffer)
		}
	}
	if err := result("end command buffer", vk.EndCommandBuffer(commandBuffer)); err != nil {
		return nil, err
	}
	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	var fence vk.Fence
	if err := result("create dispatch fence", vk.CreateFence(h.device, &fenceInfo, nil, &fence)); err != nil {
		return nil, err
	}
	defer vk.DestroyFence(h.device, fence, nil)
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{commandBuffer}}
	if err := result("submit compute work", vk.QueueSubmit(h.queue, 1, []vk.SubmitInfo{submit}, fence)); err != nil {
		return nil, err
	}
	if err := result("wait for compute work", vk.WaitForFences(h.device, 1, []vk.Fence{fence}, vk.True, vk.MaxUint64)); err != nil {
		return nil, err
	}
	output := map[string][]byte{}
	for _, allocation := range external {
		data, err := h.readAllocation(allocation)
		if err != nil {
			return nil, err
		}
		output[allocation.name] = data
	}
	return output, nil
}

func (h *vulkanHarness) allocate(name, kind string, data []byte, readback bool) (*allocatedBuffer, error) {
	buffer, memory, size, coherent, err := h.createBuffer(kind, data)
	if err != nil {
		return nil, fmt.Errorf("buffer %s: %w", name, err)
	}
	return &allocatedBuffer{name: name, data: data, readback: readback, buffer: buffer, memory: memory, size: size, coherent: coherent}, nil
}

func (h *vulkanHarness) readAllocation(allocation *allocatedBuffer) ([]byte, error) {
	data := make([]byte, len(allocation.data))
	var mapped unsafe.Pointer
	if err := result("map result buffer", vk.MapMemory(h.device, allocation.memory, 0, allocation.size, 0, &mapped)); err != nil {
		return nil, err
	}
	defer vk.UnmapMemory(h.device, allocation.memory)
	if !allocation.coherent {
		rangeInfo := vk.MappedMemoryRange{SType: vk.StructureTypeMappedMemoryRange, Memory: allocation.memory, Offset: 0, Size: vk.DeviceSize(vk.WholeSize)}
		if err := result("invalidate result buffer", vk.InvalidateMappedMemoryRanges(h.device, 1, []vk.MappedMemoryRange{rangeInfo})); err != nil {
			return nil, err
		}
	}
	copy(data, unsafe.Slice((*byte)(mapped), len(data)))
	return data, nil
}

func descriptorLayoutBinding(binding uint32, descriptorType vk.DescriptorType) vk.DescriptorSetLayoutBinding {
	return vk.DescriptorSetLayoutBinding{Binding: binding, DescriptorType: descriptorType, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageComputeBit)}
}

func descriptorWrite(set vk.DescriptorSet, binding uint32, descriptorType vk.DescriptorType, buffer vk.Buffer, size uint32) vk.WriteDescriptorSet {
	return vk.WriteDescriptorSet{SType: vk.StructureTypeWriteDescriptorSet, DstSet: set, DstBinding: binding, DescriptorCount: 1, DescriptorType: descriptorType, PBufferInfo: []vk.DescriptorBufferInfo{{Buffer: buffer, Range: vk.DeviceSize(size)}}}
}

func resolveResource(source resourceSourceMetadata, external []*allocatedBuffer, transients []transientMetadata, colors map[int]*allocatedBuffer, sizes []uint32) (*allocatedBuffer, uint32, error) {
	if source.Kind == "external" && source.Resource >= 0 && source.Resource < len(external) {
		allocation := external[source.Resource]
		return allocation, uint32(len(allocation.data)), nil
	}
	if source.Kind == "transient" && source.Resource >= 0 && source.Resource < len(transients) {
		return colors[transients[source.Resource].Color], sizes[source.Resource], nil
	}
	return nil, 0, errors.New("invalid plan resource source")
}

func recordComputeBarrier(commandBuffer vk.CommandBuffer) {
	barrier := vk.MemoryBarrier{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}
	vk.CmdPipelineBarrier(commandBuffer, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, 1, []vk.MemoryBarrier{barrier}, 0, nil, 0, nil)
}

func evaluateShape(expression shapeExpression, program *publicProgramMetadata, values []any, buffers map[string][]byte, launch [3]uint32) (uint32, error) {
	switch expression.Op {
	case "constant":
		return expression.Value, nil
	case "parameter":
		return asU32(pathValue(values[expression.Parameter], expression.Path))
	case "resourceLength":
		if expression.Resource < 0 || expression.Resource >= len(program.Resources) {
			return 0, errors.New("shape resource is invalid")
		}
		resource := program.Resources[expression.Resource]
		data := buffers[resource.Name]
		if !resource.Runtime || resource.RuntimeStride == 0 || len(data) < int(resource.RuntimeOffset) {
			return 0, errors.New("shape resource has no runtime length")
		}
		return uint32((len(data) - int(resource.RuntimeOffset)) / int(resource.RuntimeStride)), nil
	case "launchAxis":
		return launch[expression.Axis], nil
	}
	left, err := evaluateShape(*expression.Left, program, values, buffers, launch)
	if err != nil {
		return 0, err
	}
	right, err := evaluateShape(*expression.Right, program, values, buffers, launch)
	if err != nil {
		return 0, err
	}
	a, b := uint64(left), uint64(right)
	var result uint64
	switch expression.Op {
	case "add":
		result = a + b
	case "sub":
		if a < b {
			return 0, errors.New("shape underflow")
		}
		result = a - b
	case "mul":
		result = a * b
	case "div":
		if b == 0 {
			return 0, errors.New("shape division by zero")
		}
		result = a / b
	case "rem":
		if b == 0 {
			return 0, errors.New("shape remainder by zero")
		}
		result = a % b
	case "min":
		result = min(a, b)
	case "max":
		result = max(a, b)
	case "ceilDiv":
		if b == 0 {
			return 0, errors.New("ceilDiv denominator is zero")
		}
		result = (a + b - 1) / b
	default:
		return 0, errors.New("invalid shape operation")
	}
	if result > math.MaxUint32 {
		return 0, errors.New("shape overflow")
	}
	return uint32(result), nil
}

func packParameterBlock(block parameterBlockMetadata, sources []valueSourceMetadata, program *publicProgramMetadata, values []any, buffers map[string][]byte, launch [3]uint32, repeat uint32) ([]byte, error) {
	if len(sources) != len(block.Fields) {
		return nil, errors.New("parameter source count mismatch")
	}
	data := make([]byte, block.ByteSize)
	for index, field := range block.Fields {
		source := sources[index]
		var value any
		switch source.Kind {
		case "parameter":
			value = pathValue(values[source.Parameter], source.Path)
		case "bool", "i32", "u32":
			value = source.Value
		case "f32Bits":
			bits, err := asU32(source.Value)
			if err != nil {
				return nil, err
			}
			value = math.Float32frombits(bits)
		case "shape":
			evaluated, err := evaluateShape(*source.Expression, program, values, buffers, launch)
			if err != nil {
				return nil, err
			}
			value = evaluated
		case "repeat":
			value = repeat
		default:
			return nil, errors.New("invalid parameter value source")
		}
		if err := writeHostValue(data, field.ByteOffset, field.Layout, value); err != nil {
			return nil, err
		}
	}
	return data, nil
}

func writeHostValue(data []byte, offset uint32, layout hostLayout, value any) error {
	if int(offset)+4 > len(data) {
		return errors.New("parameter field is out of bounds")
	}
	switch layout.Kind {
	case "bool":
		boolean, ok := value.(bool)
		if !ok {
			return errors.New("parameter is not bool")
		}
		if boolean {
			binary.LittleEndian.PutUint32(data[offset:], 1)
		} else {
			binary.LittleEndian.PutUint32(data[offset:], 0)
		}
	case "u32":
		word, err := asU32(value)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(data[offset:], word)
	case "i32":
		word, err := asI32(value)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(data[offset:], uint32(word))
	case "f32":
		number, err := asF32(value)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(data[offset:], math.Float32bits(number))
	default:
		return fmt.Errorf("unsupported physical parameter layout %s", layout.Kind)
	}
	return nil
}

func pathValue(value any, path []string) any {
	for _, name := range path {
		if fields, ok := value.(map[string]any); ok {
			value = fields[name]
			continue
		}
		reflected := reflect.ValueOf(value)
		if reflected.Kind() == reflect.Struct {
			value = reflected.FieldByName(name).Interface()
			continue
		}
		return nil
	}
	return value
}

func asU32(value any) (uint32, error) {
	switch x := value.(type) {
	case uint32:
		return x, nil
	case int:
		if x >= 0 && uint64(x) <= math.MaxUint32 {
			return uint32(x), nil
		}
	case float64:
		if x >= 0 && x <= math.MaxUint32 && math.Trunc(x) == x {
			return uint32(x), nil
		}
	}
	return 0, errors.New("value is not uint32")
}
func asI32(value any) (int32, error) {
	switch x := value.(type) {
	case int32:
		return x, nil
	case int:
		if x >= math.MinInt32 && x <= math.MaxInt32 {
			return int32(x), nil
		}
	case float64:
		if x >= math.MinInt32 && x <= math.MaxInt32 && math.Trunc(x) == x {
			return int32(x), nil
		}
	}
	return 0, errors.New("value is not int32")
}
func asF32(value any) (float32, error) {
	switch x := value.(type) {
	case float32:
		return x, nil
	case float64:
		return float32(x), nil
	case int:
		return float32(x), nil
	}
	return 0, errors.New("value is not float32")
}

func ceilDiv(value, divisor uint32) uint32 { return 1 + (value-1)/divisor }

package spirv

import (
	"fmt"
	"slices"

	"tach/ir"
)

func validateRuntime(metadata *runtimeMetadata, logical *ir.Module) error {
	if metadata == nil || logical == nil || metadata.Vulkan != "1.3" || metadata.SPIRV != "1.6" || len(metadata.Programs) != len(logical.Programs) || !validFeatures(metadata.Features, "synchronization2", "shaderZeroInitializeWorkgroupMemory", "vulkanMemoryModel", "shaderFloat16", "storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess") || len(metadata.Features) < 3 || metadata.Features[0] != "synchronization2" || metadata.Features[1] != "shaderZeroInitializeWorkgroupMemory" || metadata.Features[2] != "vulkanMemoryModel" || len(metadata.Features) > 3 && metadata.Features[3] != "shaderFloat16" {
		return fmt.Errorf("invalid SPIR-V runtime metadata")
	}
	for i, program := range metadata.Programs {
		if program.Program != i || program.Repeat != "program" && program.Repeat != "invocation-loop" || (program.View != nil) != (logical.Programs[i].View != nil) {
			return fmt.Errorf("invalid SPIR-V program %d", i)
		}
		for _, step := range program.Steps {
			if step.Kind == "dispatch" {
				if step.Kernel < 0 || step.Kernel >= len(metadata.Kernels) {
					return fmt.Errorf("invalid SPIR-V dispatch")
				}
			} else if step.Kind != "barrier" {
				return fmt.Errorf("invalid SPIR-V step")
			}
		}
		if program.RepeatBarrier != nil && program.RepeatBarrier.Kind != "barrier" {
			return fmt.Errorf("invalid SPIR-V repeat barrier")
		}
		if program.View != nil {
			if !validView(metadata, program.View) {
				return fmt.Errorf("invalid SPIR-V view")
			}
		}
	}
	for i, kernel := range metadata.Kernels {
		if kernel.EntryPoint != fmt.Sprintf("_tach_k%d", i) || kernel.WorkgroupSize[0] == 0 || kernel.WorkgroupSize[1] == 0 || kernel.WorkgroupSize[2] == 0 {
			return fmt.Errorf("invalid SPIR-V physical kernel %d", i)
		}
		for j, binding := range kernel.Bindings {
			if binding.Group != 0 || binding.Binding != uint32(j) || binding.Kind != "buffer" {
				return fmt.Errorf("SPIR-V kernel %d bindings are not dense", i)
			}
		}
	}
	return nil
}

func validView(metadata *runtimeMetadata, view *viewMetadata) bool {
	if view.Format != "srgb8" || view.Step.Kind != "dispatch" || view.Step.Kernel < 0 || view.Step.Kernel >= len(metadata.Kernels) || view.OutputColor < 0 || view.Width.Op == "" || view.Height.Op == "" {
		return false
	}
	kernel := metadata.Kernels[view.Step.Kernel]
	if view.Output >= uint32(len(kernel.Bindings)) || kernel.Bindings[view.Output].Kind != "buffer" || slices.ContainsFunc(view.Step.Resources, func(resource resourceMetadata) bool { return resource.Binding == view.Output }) {
		return false
	}
	return view.Fused || len(kernel.Bindings) == 2 && len(view.Step.Resources) == 1 && kernel.ParameterBlock != nil && len(kernel.ParameterBlock.Fields) == 2
}

func validFeatures(features []string, allowed ...string) bool {
	last := -1
	for _, feature := range features {
		index := -1
		for i, candidate := range allowed {
			if feature == candidate {
				index = i
				break
			}
		}
		if index <= last {
			return false
		}
		last = index
	}
	return true
}

package wgsl

import (
	"fmt"

	"tach/ir"
)

func validateRuntime(metadata *runtimeMetadata, logical *ir.Module) error {
	if metadata == nil || logical == nil || len(metadata.Programs) != len(logical.Programs) || !validFeatures(metadata.Features, "shader-f16") {
		return fmt.Errorf("invalid Web runtime metadata")
	}
	for i, program := range metadata.Programs {
		if program.Program != i || program.Repeat != "program" && program.Repeat != "invocation-loop" || (program.View != nil) != (logical.Programs[i].View != nil) {
			return fmt.Errorf("invalid Web program %d", i)
		}
		for _, step := range program.Steps {
			if step.Kind != "dispatch" || step.Kernel < 0 || step.Kernel >= len(metadata.Kernels) {
				return fmt.Errorf("invalid Web dispatch")
			}
		}
		if program.View != nil {
			view := program.View
			if view.Format != "srgb8" || view.Step.Kind != "dispatch" || view.Step.Kernel < 0 || view.Step.Kernel >= len(metadata.Kernels) || view.OutputColor < 0 || view.Width.Op == "" || view.Height.Op == "" {
				return fmt.Errorf("invalid Web view")
			}
			kernel := metadata.Kernels[view.Step.Kernel]
			if view.Output >= uint32(len(kernel.Bindings)) || kernel.Bindings[view.Output].Kind != "texture" {
				return fmt.Errorf("invalid Web view output")
			}
			for _, resource := range view.Step.Resources {
				if resource.Binding == view.Output {
					return fmt.Errorf("web view output is also an input")
				}
			}
			if !view.Fused && (len(kernel.Bindings) != 2 || len(view.Step.Resources) != 1 || kernel.ParameterBlock == nil || len(kernel.ParameterBlock.Fields) != 2) {
				return fmt.Errorf("invalid standalone Web view projection")
			}
		}
	}
	for i, kernel := range metadata.Kernels {
		if kernel.EntryPoint != fmt.Sprintf("_tach_k%d", i) || kernel.WorkgroupSize[0] == 0 || kernel.WorkgroupSize[1] == 0 || kernel.WorkgroupSize[2] == 0 {
			return fmt.Errorf("invalid Web physical kernel %d", i)
		}
		for j, binding := range kernel.Bindings {
			if binding.Group != 0 || binding.Binding != uint32(j) || binding.Kind != "buffer" && binding.Kind != "texture" {
				return fmt.Errorf("web kernel %d bindings are not dense", i)
			}
		}
	}
	return nil
}

func validFeatures(features []string, allowed ...string) bool {
	return len(features) == 0 || len(features) == 1 && len(allowed) == 1 && features[0] == allowed[0]
}

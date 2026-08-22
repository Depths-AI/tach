package spirv

import (
	"fmt"
	"tach/ir"
)

func verify(executable *plan) error {
	if executable == nil || executable.Logical == nil || executable.KernelModule == nil {
		return fmt.Errorf("incomplete executable")
	}
	if err := ir.Verify(executable.Logical); err != nil {
		return fmt.Errorf("logical module: %w", err)
	}
	if err := ir.VerifyKernel(executable.KernelModule); err != nil {
		return fmt.Errorf("physical kernel module: %w", err)
	}
	entries := map[string]bool{}
	for i, kernel := range executable.PhysicalKernels {
		if kernel.Entry != ir.PrivateEntryName(i) || kernel.Function == nil || kernel.Function.Name != kernel.Entry || entries[kernel.Entry] {
			return fmt.Errorf("physical kernel %d has invalid private entry", i)
		}
		entries[kernel.Entry] = true
		if kernel.Workgroup[0] == 0 || kernel.Workgroup[1] == 0 || kernel.Workgroup[2] == 0 || len(kernel.Bindings) != len(kernel.Function.BufferParams) {
			return fmt.Errorf("physical kernel %s has invalid workgroup/bindings", kernel.Entry)
		}
		for binding, descriptor := range kernel.Bindings {
			if descriptor.Buffer != binding || descriptor.Binding != uint32(binding) {
				return fmt.Errorf("physical kernel %s bindings are not dense", kernel.Entry)
			}
		}
		if kernel.Projection && (kernel.FusedView || len(kernel.Bindings) != 2) {
			return fmt.Errorf("physical projection kernel %s is invalid", kernel.Entry)
		}
		if kernel.FusedView {
			if kernel.Projection || kernel.ViewBinding < 0 || kernel.ViewBinding >= len(kernel.Bindings) {
				return fmt.Errorf("physical fused view kernel %s is invalid", kernel.Entry)
			}
		}
	}
	if len(executable.Programs) != len(executable.Logical.Programs) {
		return fmt.Errorf("target program count mismatch")
	}
	for i, plan := range executable.Programs {
		if plan.Program != i || plan.Repeat == 0 {
			return fmt.Errorf("program plan %d is invalid", i)
		}
		for _, step := range plan.Steps {
			if step.Kind == dispatchStepKind && (step.Kernel < 0 || step.Kernel >= len(executable.PhysicalKernels)) {
				return fmt.Errorf("program plan %d references invalid kernel", i)
			}
		}
		logical := executable.Logical.Programs[i].View
		if (plan.View != nil) != (logical != nil) {
			return fmt.Errorf("program plan %d view contract mismatch", i)
		}
		if plan.View != nil {
			view := plan.View
			if view.step.Kind != dispatchStepKind || view.step.Kernel < 0 || view.step.Kernel >= len(executable.PhysicalKernels) || view.OutputColor < 0 || view.Width != logical.Width || view.Height != logical.Height {
				return fmt.Errorf("program plan %d has invalid view", i)
			}
			kernel := executable.PhysicalKernels[view.step.Kernel]
			if view.Output >= uint32(len(kernel.Bindings)) || view.Fused && (!kernel.FusedView || int(view.Output) != kernel.ViewBinding) || !view.Fused && (!kernel.Projection || view.Output != 1 || len(view.step.Resources) != 1) {
				return fmt.Errorf("program plan %d has invalid view kernel", i)
			}
			for _, resource := range view.step.Resources {
				if resource.Binding == view.Output {
					return fmt.Errorf("program plan %d binds the view output as input", i)
				}
			}
			for _, transient := range plan.Transients {
				if transient.Color == view.OutputColor {
					return fmt.Errorf("program plan %d aliases view output and transient", i)
				}
			}
		}
	}
	return nil
}

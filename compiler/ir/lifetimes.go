package ir

type TransientLifetime struct {
	Resource  ResourceID
	Color     int
	FirstStep int
	LastStep  int
}

func AnalyzeTransientLifetimes(program *Program, omitted ResourceID) []TransientLifetime {
	var out []TransientLifetime
	for _, resource := range program.Resources {
		if resource.Kind != TransientResourceKind || resource.ID == omitted {
			continue
		}
		first, last := len(program.Dispatches), -1
		for i, dispatch := range program.Dispatches {
			for _, argument := range dispatch.Buffers {
				if argument.Resource == resource.ID {
					first, last = min(first, i), max(last, i)
				}
			}
		}
		if program.View != nil && program.View.Source == resource.ID {
			last = len(program.Dispatches)
		}
		color := 0
		for overlaps(out, color, first, last) {
			color++
		}
		out = append(out, TransientLifetime{Resource: resource.ID, Color: color, FirstStep: first, LastStep: last})
	}
	return out
}

func overlaps(allocations []TransientLifetime, color, first, last int) bool {
	for _, prior := range allocations {
		if prior.Color == color && first <= prior.LastStep && prior.FirstStep <= last {
			return true
		}
	}
	return false
}

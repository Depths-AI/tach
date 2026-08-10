package spirv

import (
	"fmt"
	"strings"
)

// Disassemble decodes and validates a Pine SPIR-V module, then prints a stable
// textual form intended for compiler debugging and golden tests.
func Disassemble(data []byte) (string, error) {
	m, err := Decode(data)
	if err != nil {
		return "", err
	}
	if err := Validate(data); err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "; SPIR-V 1.3\n; Bound: %d\n", m.Bound)
	for _, in := range m.Instructions {
		fmt.Fprintf(&b, "%04d: %-24s", in.Offset, opName(in.Op))
		writeOperands(&b, in)
		b.WriteByte('\n')
	}
	return b.String(), nil
}

func writeOperands(b *strings.Builder, in Instruction) {
	a := in.Operands
	switch in.Op {
	case OpName:
		name, _, _ := literalString(a, 1)
		fmt.Fprintf(b, " %%%d %q", a[0], name)
	case OpMemberName:
		name, _, _ := literalString(a, 2)
		fmt.Fprintf(b, " %%%d %d %q", a[0], a[1], name)
	case OpExtInstImport:
		name, _, _ := literalString(a, 1)
		fmt.Fprintf(b, " %%%d %q", a[0], name)
	case OpExtInst:
		fmt.Fprintf(b, " %%%d %%%d %%%d %d", a[0], a[1], a[2], a[3])
		for _, id := range a[4:] {
			fmt.Fprintf(b, " %%%d", id)
		}
	case OpDot:
		fmt.Fprintf(b, " %%%d %%%d %%%d %%%d", a[0], a[1], a[2], a[3])
	case OpEntryPoint:
		name, next, _ := literalString(a, 2)
		fmt.Fprintf(b, " GLCompute %%%d %q", a[1], name)
		for _, id := range a[next:] {
			fmt.Fprintf(b, " %%%d", id)
		}
	default:
		for _, x := range a {
			fmt.Fprintf(b, " %d", x)
		}
	}
}

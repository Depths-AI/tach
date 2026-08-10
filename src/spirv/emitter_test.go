package spirv_test

import (
	"os"
	"strings"
	"testing"

	"tach/src/parser"
	"tach/src/sema"
	"tach/src/spirv"
)

func emitSource(t *testing.T, name, source string) []byte {
	t.Helper()
	a, err := parser.Parse(name, source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := spirv.Emit(m)
	if err != nil {
		t.Fatal(err)
	}
	return bin
}

func TestParticlesSPIRV(t *testing.T) {
	src, err := os.ReadFile("../../examples/particles.tach")
	if err != nil {
		t.Fatal(err)
	}
	a, err := parser.Parse("particles.tach", string(src))
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := spirv.Emit(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(bin) < 20 {
		t.Fatalf("short SPIR-V binary: %d", len(bin))
	}
	s, err := spirv.Summary(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(s, "entries [integrate]") {
		t.Fatalf("unexpected summary: %s", s)
	}
}

func TestWorkgroupAggregatesHaveNoExplicitLayout(t *testing.T) {
	tests := []struct {
		name   string
		source string
	}{
		{
			name: "array",
			source: `
@workgroupSize(1)
export compute arrayMemory(out: storage<u32, read_write>) {
  workgroup scratch: u32[4];
  scratch[0u] = 7u;
  out = scratch[0u];
}`,
		},
		{
			name: "struct",
			source: `
type Pair = { x: u32, y: u32 };
@workgroupSize(1)
export compute structMemory(out: storage<u32, read_write>) {
  workgroup pair: Pair;
  pair = { x: 7u, y: 9u };
  workgroupBarrier();
  const copy: Pair = pair;
  out = copy.x + copy.y;
}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bin := emitSource(t, tc.name+".tach", tc.source)
			m, err := spirv.Decode(bin)
			if err != nil {
				t.Fatal(err)
			}

			pointerElem := map[uint32]uint32{}
			children := map[uint32][]uint32{}
			explicit := map[uint32]string{}
			var workgroupPointers []uint32
			for _, in := range m.Instructions {
				a := in.Operands
				switch in.Op {
				case spirv.OpTypePointer:
					pointerElem[a[0]] = a[2]
				case spirv.OpTypeVector, spirv.OpTypeArray, spirv.OpTypeRuntimeArray:
					children[a[0]] = []uint32{a[1]}
				case spirv.OpTypeStruct:
					children[a[0]] = append([]uint32(nil), a[1:]...)
				case spirv.OpVariable:
					if a[2] == spirv.StorageWorkgroup {
						workgroupPointers = append(workgroupPointers, a[0])
					}
				case spirv.OpDecorate:
					switch a[1] {
					case spirv.DecorationBlock:
						explicit[a[0]] = "Block"
					case spirv.DecorationArrayStride:
						explicit[a[0]] = "ArrayStride"
					}
				case spirv.OpMemberDecorate:
					if a[2] == spirv.DecorationOffset {
						explicit[a[0]] = "Offset"
					}
				}
			}
			if len(workgroupPointers) != 1 {
				t.Fatalf("found %d Workgroup variables, want 1", len(workgroupPointers))
			}

			seen := map[uint32]bool{}
			var walk func(uint32)
			walk = func(id uint32) {
				if seen[id] {
					return
				}
				seen[id] = true
				if decoration := explicit[id]; decoration != "" {
					t.Errorf("Workgroup-reachable type %%%d carries %s", id, decoration)
				}
				for _, child := range children[id] {
					walk(child)
				}
			}
			walk(pointerElem[workgroupPointers[0]])
		})
	}
}

func TestHostResourceAggregatesKeepExplicitLayout(t *testing.T) {
	src, err := os.ReadFile("../../examples/particles.tach")
	if err != nil {
		t.Fatal(err)
	}
	bin := emitSource(t, "particles.tach", string(src))
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var block, offset, stride, workgroupPrologue bool
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpDecorate:
			block = block || a[1] == spirv.DecorationBlock
			stride = stride || a[1] == spirv.DecorationArrayStride
			workgroupPrologue = workgroupPrologue || a[1] == spirv.DecorationBuiltIn && a[2] == spirv.BuiltInLocalInvocationIndex
		case spirv.OpMemberDecorate:
			offset = offset || a[2] == spirv.DecorationOffset
		case spirv.OpConstantNull:
			workgroupPrologue = true
		}
	}
	if !block || !offset || !stride {
		t.Fatalf("host resource layout decorations: Block=%v Offset=%v ArrayStride=%v", block, offset, stride)
	}
	if workgroupPrologue {
		t.Fatal("kernel without Workgroup variables received an initialization prologue")
	}
}

func TestSameStructCrossesHostAndWorkgroupRepresentations(t *testing.T) {
	bin := emitSource(t, "shared-struct.tach", `
type Pair = { x: u32, y: u32 };
@workgroupSize(1)
export compute sharedStruct(io: storage<Pair, read_write>) {
  workgroup pair: Pair;
  pair = io;
  io = pair;
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}

	pointerElem := map[uint32]uint32{}
	structMembers := map[uint32][]uint32{}
	offsetCount := map[uint32]int{}
	var workgroupType, hostType uint32
	var hasLogicalConstruct, hasLogicalExtract bool
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpTypePointer:
			pointerElem[a[0]] = a[2]
		case spirv.OpTypeStruct:
			structMembers[a[0]] = append([]uint32(nil), a[1:]...)
		case spirv.OpMemberDecorate:
			if a[2] == spirv.DecorationOffset {
				offsetCount[a[0]]++
			}
		case spirv.OpVariable:
			switch a[2] {
			case spirv.StorageWorkgroup:
				workgroupType = pointerElem[a[0]]
			case spirv.StorageStorageBuffer:
				wrapper := pointerElem[a[0]]
				members := structMembers[wrapper]
				if len(members) == 1 {
					hostType = members[0]
				}
			}
		}
	}
	if workgroupType == 0 || hostType == 0 {
		t.Fatalf("representation roots: Workgroup=%%%d host=%%%d", workgroupType, hostType)
	}
	if workgroupType == hostType {
		t.Fatalf("host and Workgroup aggregates reused decorated type %%%d", hostType)
	}
	if offsetCount[workgroupType] != 0 || offsetCount[hostType] != 2 {
		t.Fatalf("member offsets: Workgroup=%d host=%d, want 0 and 2", offsetCount[workgroupType], offsetCount[hostType])
	}
	for _, in := range m.Instructions {
		switch in.Op {
		case spirv.OpCompositeConstruct:
			hasLogicalConstruct = hasLogicalConstruct || in.Operands[0] == workgroupType
		case spirv.OpCompositeExtract:
			hasLogicalExtract = hasLogicalExtract || in.Operands[0] == structMembers[workgroupType][0]
		}
	}
	if !hasLogicalConstruct || !hasLogicalExtract {
		t.Fatalf("structural host boundary: construct=%v extract=%v", hasLogicalConstruct, hasLogicalExtract)
	}
}

func TestWorkgroupMemoryIsZeroInitialized(t *testing.T) {
	bin := emitSource(t, "zero-workgroup.tach", `
@workgroupSize(1)
export compute zeroWorkgroup(out: storage<u32, read_write>) {
  workgroup scratch: u32;
  out = scratch;
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	workgroupVars := map[uint32]bool{}
	nulls := map[uint32]bool{}
	hasBarrier := false
	hasLocalIndex := false
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpVariable:
			if a[2] == spirv.StorageWorkgroup {
				workgroupVars[a[1]] = true
			}
		case spirv.OpConstantNull:
			nulls[a[1]] = true
		case spirv.OpControlBarrier:
			hasBarrier = true
		case spirv.OpDecorate:
			hasLocalIndex = hasLocalIndex || a[1] == spirv.DecorationBuiltIn && a[2] == spirv.BuiltInLocalInvocationIndex
		}
	}
	hasZeroStore := false
	for _, in := range m.Instructions {
		if in.Op == spirv.OpStore && workgroupVars[in.Operands[0]] && nulls[in.Operands[1]] {
			hasZeroStore = true
			break
		}
	}
	if len(workgroupVars) != 1 || !hasZeroStore || !hasBarrier || !hasLocalIndex {
		t.Fatalf("zero prologue: WorkgroupVars=%d nullStore=%v barrier=%v localIndex=%v", len(workgroupVars), hasZeroStore, hasBarrier, hasLocalIndex)
	}
}

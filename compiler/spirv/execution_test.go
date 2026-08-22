package spirv_test

import (
	"fmt"
	"reflect"
	"tach/spirv"
	"testing"
)

func entryName(operands []uint32) (string, int, error) {
	bytes := make([]byte, 0, 16)
	for index, word := range operands[2:] {
		for shift := range 4 {
			value := byte(word >> (8 * shift))
			if value == 0 {
				return string(bytes), index + 3, nil
			}
			bytes = append(bytes, value)
		}
	}
	return "", 0, fmt.Errorf("unterminated entry name")
}

func TestHelpersRequestConstInlining(t *testing.T) {
	bin := emitSource(t, "inline.tach", `
function twice(x: float32): float32 { return x + x; }
@workgroup(1)
export function useHelper[i](out: buffer<float32>) { out = twice(2.0); }
`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var controls []uint32
	for _, in := range m.Instructions {
		if in.Op == spirv.OpFunction {
			controls = append(controls, in.Operands[2])
		}
	}
	want := []uint32{spirv.FunctionControlInline | spirv.FunctionControlConst, spirv.FunctionControlNone}
	if !reflect.DeepEqual(controls, want) {
		t.Fatalf("function controls %v, want %v", controls, want)
	}
}

func TestVulkan13MemorySemantics(t *testing.T) {
	bin := emitSource(t, "semantics.tach", `
@workgroup(1)
export function semantics[i](out: buffer<uint32[]>, counters: buffer<atomic<uint32>[]>, add: uint32) {
  let scratch: shared<uint32>;
  scratch = add;
  workgroupBarrier();
  if (i < out.length) { out[i] = scratch; }
  if (i < counters.length) { atomicAdd(counters[i], 1); }
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	caps := map[uint32]int{}
	pointerStorage := map[uint32]uint32{}
	valueType := map[uint32]uint32{}
	var memoryModel []uint32
	var loads, stores, atomics, barriers int
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpCapability:
			caps[a[0]]++
		case spirv.OpMemoryModel:
			memoryModel = a
		case spirv.OpTypePointer:
			pointerStorage[a[0]] = a[1]
		case spirv.OpVariable, spirv.OpAccessChain, spirv.OpFunctionParameter:
			if len(a) >= 2 {
				valueType[a[1]] = a[0]
			}
		case spirv.OpLoad:
			valueType[a[1]] = a[0]
			loads++
			storage := pointerStorage[valueType[a[2]]]
			if a[3]&spirv.MemoryAccessAligned == 0 || a[4] == 0 || a[4]&(a[4]-1) != 0 {
				t.Fatalf("load missing Aligned: %v", a)
			}
			if storage == spirv.StorageInput {
				if a[3]&spirv.MemoryAccessNonPrivatePointer != 0 {
					t.Fatalf("input load is NonPrivate: %v", a)
				}
			} else if a[3]&spirv.MemoryAccessNonPrivatePointer == 0 {
				t.Fatalf("shared load missing NonPrivatePointer: %v", a)
			}
		case spirv.OpStore:
			stores++
			if a[2]&spirv.MemoryAccessAligned == 0 || a[2]&spirv.MemoryAccessNonPrivatePointer == 0 {
				t.Fatalf("store missing Aligned|NonPrivatePointer: %v", a)
			}
		case spirv.OpAtomicIAdd:
			atomics++
			scope := uint32(0)
			for _, inst := range m.Instructions {
				if inst.Op == spirv.OpConstant && inst.Operands[1] == a[3] {
					scope = inst.Operands[2]
				}
			}
			storage := pointerStorage[valueType[a[2]]]
			want := spirv.ScopeQueueFamily
			if storage == spirv.StorageWorkgroup {
				want = spirv.ScopeWorkgroup
			}
			if scope != want {
				t.Fatalf("atomic scope=%d storage=%d, want %d", scope, storage, want)
			}
		case spirv.OpControlBarrier:
			barriers++
			sem := uint32(0)
			for _, inst := range m.Instructions {
				if inst.Op == spirv.OpConstant && inst.Operands[1] == a[2] {
					sem = inst.Operands[2]
				}
			}
			need := spirv.MemorySemanticsAcquireRelease | spirv.MemorySemanticsMakeAvailable | spirv.MemorySemanticsMakeVisible
			if sem&need != need {
				t.Fatalf("barrier semantics=0x%x missing availability/visibility", sem)
			}
		}
	}
	if caps[spirv.CapabilityShader] != 1 || caps[spirv.CapabilityVulkanMemoryModel] != 1 || len(caps) != 2 {
		t.Fatalf("capabilities = %v", caps)
	}
	if len(memoryModel) != 2 || memoryModel[0] != spirv.AddressingLogical || memoryModel[1] != spirv.MemoryVulkan {
		t.Fatalf("memory model = %v", memoryModel)
	}
	if loads == 0 || stores == 0 || atomics == 0 || barriers == 0 {
		t.Fatalf("coverage loads=%d stores=%d atomics=%d barriers=%d", loads, stores, atomics, barriers)
	}
}

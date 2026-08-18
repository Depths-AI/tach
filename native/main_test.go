//go:build tachvulkan

package main

import (
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

type wireWriter []byte

func (w *wireWriter) u32(value uint32) { *w = binary.LittleEndian.AppendUint32(*w, value) }
func (w *wireWriter) raw(value string) {
	w.u32(uint32(len(value)))
	*w = append(*w, value...)
	for len(*w)%4 != 0 {
		*w = append(*w, 0)
	}
}

func validModuleWire() []byte {
	var wire wireWriter
	wire.u32(nativeABI)
	wire.u32(1)
	wire.raw("kernel")
	wire.u32(2)
	wire.u32(0)
	wire.u32(4)
	wire.u32(1)
	wire.u32(16)
	wire.u32(1)
	wire.u32(2)
	wire.u32(16)
	return wire
}

func TestModuleWire(t *testing.T) {
	kernels, err := parseModule(reader{data: validModuleWire()})
	if err != nil || len(kernels) != 1 || kernels[0].entry != "kernel" || len(kernels[0].bindings) != 2 || kernels[0].parameterBinding != 2 {
		t.Fatalf("parseModule() = %#v, %v", kernels, err)
	}
	for name, mutate := range map[string]func([]byte){
		"version":           func(wire []byte) { wire[0]++ },
		"duplicate binding": func(wire []byte) { binary.LittleEndian.PutUint32(wire[32:], 0) },
		"trailing data":     func(wire []byte) { wire = append(wire, 0); _ = wire },
	} {
		t.Run(name, func(t *testing.T) {
			wire := append([]byte(nil), validModuleWire()...)
			if name == "trailing data" {
				wire = append(wire, 0)
			} else {
				mutate(wire)
			}
			if _, err := parseModule(reader{data: wire}); err == nil {
				t.Fatal("malformed module wire was accepted")
			}
		})
	}
}

func TestBatchWire(t *testing.T) {
	var wire wireWriter
	wire.u32(nativeABI)
	wire.u32(1) // version, commands
	wire.u32(1)
	wire.u32(2)
	wire.u32(1)
	wire.u32(0)
	wire.u32(64) // module, repeat, scratch
	wire.u32(1)
	wire.u32(0)
	wire.u32(0)
	wire.u32(1)
	wire.u32(1)
	wire.u32(1) // steps, dispatch, kernel, groups
	wire.u32(1)
	wire.u32(0)
	wire.u32(1)
	wire.u32(0)
	wire.u32(64) // external resource
	wire.raw("")
	wire.u32(1)
	wire.u32(1)
	wire.u32(0) // parameters, repeat barrier scratch
	commands, err := parseBatch(reader{data: wire})
	if err != nil || len(commands) != 1 || commands[0].repeat != 2 || len(commands[0].steps) != 1 || commands[0].steps[0].groups != [3]uint32{1, 1, 1} {
		t.Fatalf("parseBatch() = %#v, %v", commands, err)
	}
	for _, offset := range []int{0, 4, 12, 40} {
		broken := append([]byte(nil), wire...)
		binary.LittleEndian.PutUint32(broken[offset:], 0)
		if _, err := parseBatch(reader{data: broken}); err == nil {
			t.Fatalf("malformed batch at byte %d was accepted", offset)
		}
	}
}

func TestVulkan13Contract(t *testing.T) {
	source, err := os.ReadFile("vulkan.c")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, required := range []string{"VK_API_VERSION_1_3", "shaderZeroInitializeWorkgroupMemory", "synchronization2", "vulkanMemoryModel", "X(CmdPipelineBarrier2)", "X(QueueSubmit2)", "VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT"} {
		if !strings.Contains(text, required) {
			t.Errorf("native runtime lacks %s", required)
		}
	}
	for _, obsolete := range []string{"vkCmdPipelineBarrier\"", "vkQueueSubmit\"", "VK_API_VERSION_1_1", "VK_API_VERSION_1_2"} {
		if strings.Contains(text, obsolete) {
			t.Errorf("native runtime retains %s", obsolete)
		}
	}
}

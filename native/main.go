//go:build tachvulkan

package main

/*
#cgo CFLAGS: -std=c11
#cgo linux LDFLAGS: -ldl
#include <stdlib.h>
#include "vulkan.h"
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime/cgo"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"
)

const nativeABI = 3

type nativeModule struct {
	value   *C.tv_module
	kernels []kernelDescription
}

type scratchKey struct{ module, color uint32 }
type nativeBuffer struct {
	value *C.tv_buffer
	size  uint32
}

type session struct {
	sync.Mutex
	context     *C.tv_context
	handle      cgo.Handle
	modules     map[uint32]nativeModule
	buffers     map[uint32]nativeBuffer
	scratch     map[scratchKey]nativeBuffer
	submissions map[uint64]*C.tv_submission
	available   []*C.tv_submission
	nextModule  uint32
	nextBuffer  uint32
	nextSubmit  uint64
	err         string
}

var globalError struct {
	sync.Mutex
	text string
}

func setGlobalError(err error) {
	globalError.Lock()
	globalError.text = err.Error()
	globalError.Unlock()
}

func boolInt(value bool) C.int {
	if value {
		return 1
	}
	return 0
}

func (s *session) fail(err error) C.int32_t {
	if err != nil {
		s.err = err.Error()
	}
	return -1
}

func (s *session) nativeError() error { return errors.New(C.GoString(C.tv_error(s.context))) }

func resolve(pointer unsafe.Pointer) (resolved *session) {
	if pointer == nil {
		return nil
	}
	defer func() {
		if recover() != nil {
			resolved = nil
		}
	}()
	resolved, _ = cgo.Handle(uintptr(pointer)).Value().(*session)
	return resolved
}

type reader struct {
	data   []byte
	offset int
}

func newReader(pointer *C.uint8_t, length C.size_t) reader {
	if pointer == nil || length == 0 {
		return reader{}
	}
	return reader{data: C.GoBytes(unsafe.Pointer(pointer), C.int(length))}
}

func (r *reader) u32() (uint32, error) {
	if r.offset+4 > len(r.data) {
		return 0, errors.New("truncated native wire value")
	}
	value := uint32(r.data[r.offset]) | uint32(r.data[r.offset+1])<<8 | uint32(r.data[r.offset+2])<<16 | uint32(r.data[r.offset+3])<<24
	r.offset += 4
	return value, nil
}

func (r *reader) raw() ([]byte, error) {
	length, err := r.u32()
	if err != nil || uint64(r.offset)+uint64(length) > uint64(len(r.data)) {
		return nil, errors.New("truncated native wire bytes")
	}
	value := r.data[r.offset : r.offset+int(length)]
	r.offset += int(length)
	for r.offset%4 != 0 {
		r.offset++
		if r.offset > len(r.data) {
			return nil, errors.New("truncated native wire padding")
		}
	}
	return value, nil
}

func (r *reader) text() (string, error) {
	value, err := r.raw()
	if err != nil {
		return "", err
	}
	if !utf8.Valid(value) || strings.IndexByte(string(value), 0) >= 0 {
		return "", errors.New("native wire text is not valid UTF-8")
	}
	return string(value), nil
}

func (r *reader) end() error {
	if r.offset != len(r.data) {
		return errors.New("native wire has trailing data")
	}
	return nil
}

type kernelDescription struct {
	entry             string
	bindings          []bindingDescription
	parameterBinding  uint32
	parameterByteSize uint32
}
type bindingDescription struct{ binding, minimumByteSize uint32 }

func parseModule(r reader) (uint32, []kernelDescription, error) {
	version, err := r.u32()
	if err != nil || version != nativeABI {
		return 0, nil, errors.New("unsupported native module wire version")
	}
	features, err := r.u32()
	if err != nil || features > 7 || features&6 != 0 && features&1 == 0 {
		return 0, nil, errors.New("invalid native module features")
	}
	count, err := r.u32()
	if err != nil || count == 0 || count > 1<<20 {
		return 0, nil, errors.New("invalid physical kernel count")
	}
	kernels := make([]kernelDescription, count)
	for index := range kernels {
		entry, err := r.text()
		if err != nil || entry == "" {
			return 0, nil, errors.New("invalid physical kernel entry point")
		}
		bindingCount, err := r.u32()
		if err != nil || bindingCount == 0 || bindingCount > 1<<20 {
			return 0, nil, fmt.Errorf("kernel %s has an invalid binding count", entry)
		}
		bindings := make([]bindingDescription, bindingCount)
		seen := map[uint32]bool{}
		for binding := range bindings {
			bindings[binding].binding, err = r.u32()
			if err == nil {
				bindings[binding].minimumByteSize, err = r.u32()
			}
			if err != nil || seen[bindings[binding].binding] {
				return 0, nil, fmt.Errorf("kernel %s has invalid bindings", entry)
			}
			seen[bindings[binding].binding] = true
		}
		hasParameters, err := r.u32()
		if err != nil || hasParameters > 1 {
			return 0, nil, fmt.Errorf("kernel %s has an invalid parameter declaration", entry)
		}
		kernels[index] = kernelDescription{entry: entry, bindings: bindings}
		if hasParameters == 1 {
			kernels[index].parameterBinding, err = r.u32()
			if err == nil {
				kernels[index].parameterByteSize, err = r.u32()
			}
			if err != nil || kernels[index].parameterByteSize == 0 || seen[kernels[index].parameterBinding] {
				return 0, nil, fmt.Errorf("kernel %s has an invalid parameter block", entry)
			}
		}
	}
	return features, kernels, r.end()
}

type wireResource struct {
	binding uint32
	kind    uint32
	id      uint32
	size    uint32
}
type wireStep struct {
	barrier    bool
	kernel     uint32
	groups     [3]uint32
	resources  []wireResource
	parameters []byte
}
type wireCommand struct {
	module        uint32
	repeat        uint32
	scratch       map[uint32]uint32
	steps         []wireStep
	repeatBarrier []wireResource
}

func parseResource(r *reader, binding bool) (wireResource, error) {
	var resource wireResource
	var err error
	if binding {
		resource.binding, err = r.u32()
	}
	if err == nil {
		resource.kind, err = r.u32()
	}
	if err == nil {
		resource.id, err = r.u32()
	}
	if binding && err == nil {
		resource.size, err = r.u32()
	}
	if err != nil || resource.kind > 1 || binding && resource.size == 0 {
		return wireResource{}, errors.New("invalid native command resource")
	}
	return resource, nil
}

func parseBatch(r reader) ([]wireCommand, error) {
	version, err := r.u32()
	if err != nil || version != nativeABI {
		return nil, errors.New("unsupported native batch wire version")
	}
	count, err := r.u32()
	if err != nil || count == 0 || count > 1<<20 {
		return nil, errors.New("invalid native command count")
	}
	commands := make([]wireCommand, count)
	for commandIndex := range commands {
		command := &commands[commandIndex]
		command.module, err = r.u32()
		if err == nil {
			command.repeat, err = r.u32()
		}
		if err != nil || command.repeat == 0 {
			return nil, errors.New("invalid native command header")
		}
		scratchCount, err := r.u32()
		if err != nil || scratchCount > 1<<20 {
			return nil, errors.New("invalid native scratch count")
		}
		command.scratch = make(map[uint32]uint32, scratchCount)
		for range scratchCount {
			color, colorErr := r.u32()
			bytes, bytesErr := r.u32()
			if colorErr != nil || bytesErr != nil || bytes == 0 || command.scratch[color] != 0 {
				return nil, errors.New("invalid native scratch allocation")
			}
			command.scratch[color] = bytes
		}
		stepCount, err := r.u32()
		if err != nil || stepCount == 0 || stepCount > 1<<20 {
			return nil, errors.New("invalid native step count")
		}
		command.steps = make([]wireStep, stepCount)
		for stepIndex := range command.steps {
			step := &command.steps[stepIndex]
			kind, kindErr := r.u32()
			if kindErr != nil || kind > 1 {
				return nil, errors.New("invalid native step kind")
			}
			step.barrier = kind == 1
			if !step.barrier {
				step.kernel, err = r.u32()
				for axis := range step.groups {
					if err == nil {
						step.groups[axis], err = r.u32()
					}
					if step.groups[axis] == 0 {
						err = errors.New("zero dispatch group")
					}
				}
			}
			resourceCount, countErr := r.u32()
			if err != nil || countErr != nil || resourceCount > 1<<20 {
				return nil, errors.New("invalid native step")
			}
			step.resources = make([]wireResource, resourceCount)
			for index := range step.resources {
				step.resources[index], err = parseResource(&r, !step.barrier)
				if err != nil {
					return nil, err
				}
			}
			if !step.barrier {
				step.parameters, err = r.raw()
				if err != nil {
					return nil, err
				}
			}
		}
		repeatCount, err := r.u32()
		if err != nil || repeatCount > 1<<20 {
			return nil, errors.New("invalid repeat barrier")
		}
		command.repeatBarrier = make([]wireResource, repeatCount)
		for index := range command.repeatBarrier {
			command.repeatBarrier[index], err = parseResource(&r, false)
			if err != nil {
				return nil, err
			}
		}
	}
	return commands, r.end()
}

func copyOutput(output *C.uint8_t, capacity C.size_t, value []byte) C.int32_t {
	if output == nil || uint64(capacity) < uint64(len(value)) {
		return -1
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(output)), len(value)), value)
	return C.int32_t(len(value))
}

//export tach_abi_version
func tach_abi_version() C.uint32_t { return nativeABI }

//export tach_open
func tach_open(options *C.uint8_t, length C.size_t, output *unsafe.Pointer) C.int32_t {
	if output == nil {
		setGlobalError(errors.New("session output pointer is required"))
		return -1
	}
	var requested struct {
		PowerPreference string `json:"powerPreference"`
	}
	decoder := json.NewDecoder(strings.NewReader(string(newReader(options, length).data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&requested); err != nil || requested.PowerPreference != "" && requested.PowerPreference != "low-power" && requested.PowerPreference != "high-performance" {
		if err == nil {
			err = errors.New("powerPreference must be low-power or high-performance")
		}
		setGlobalError(err)
		return -1
	}
	context := C.tv_open(boolInt(requested.PowerPreference == "high-performance"))
	if context == nil {
		setGlobalError(errors.New(C.GoString(C.tv_global_error())))
		return -1
	}
	s := &session{context: context, modules: map[uint32]nativeModule{}, buffers: map[uint32]nativeBuffer{}, scratch: map[scratchKey]nativeBuffer{}, submissions: map[uint64]*C.tv_submission{}, nextModule: 1, nextBuffer: 1, nextSubmit: 1}
	s.handle = cgo.NewHandle(s)
	*output = unsafe.Pointer(uintptr(s.handle))
	return 0
}

//export tach_info
func tach_info(pointer unsafe.Pointer, output *C.uint8_t, capacity C.size_t) C.int32_t {
	s := resolve(pointer)
	if s == nil {
		return -1
	}
	s.Lock()
	defer s.Unlock()
	types := []string{"unknown", "integrated", "discrete", "virtual", "cpu"}
	typeIndex := int(C.tv_device_type(s.context))
	if typeIndex < 0 || typeIndex >= len(types) {
		typeIndex = 0
	}
	value, err := json.Marshal(map[string]string{
		"backend": "vulkan", "name": C.GoString(C.tv_device_name(s.context)),
		"vendor": fmt.Sprintf("0x%04x", uint32(C.tv_vendor_id(s.context))), "type": types[typeIndex],
	})
	if err != nil {
		return s.fail(err)
	}
	result := copyOutput(output, capacity, value)
	if result < 0 {
		return s.fail(errors.New("adapter information output is too small"))
	}
	return result
}

//export tach_module
func tach_module(pointer unsafe.Pointer, spirv *C.uint8_t, spirvLength C.size_t, description *C.uint8_t, descriptionLength C.size_t, output *C.uint32_t) C.int32_t {
	s := resolve(pointer)
	if s == nil || output == nil {
		return -1
	}
	features, kernels, err := parseModule(newReader(description, descriptionLength))
	if err != nil {
		s.Lock()
		defer s.Unlock()
		return s.fail(err)
	}
	spirvBytes := newReader(spirv, spirvLength).data
	s.Lock()
	defer s.Unlock()
	var spirvPointer *C.uint8_t
	if len(spirvBytes) > 0 {
		spirvPointer = (*C.uint8_t)(unsafe.Pointer(&spirvBytes[0]))
	}
	module := C.tv_module_create(s.context, spirvPointer, C.size_t(len(spirvBytes)), C.uint32_t(features), C.uint32_t(len(kernels)))
	if module == nil {
		return s.fail(s.nativeError())
	}
	for index, kernel := range kernels {
		bindings := make([]C.tv_binding, len(kernel.bindings))
		for binding := range bindings {
			bindings[binding].binding = C.uint32_t(kernel.bindings[binding].binding)
			bindings[binding].minimum_size = C.uint32_t(kernel.bindings[binding].minimumByteSize)
		}
		entry := C.CString(kernel.entry)
		status := C.tv_module_kernel(module, C.uint32_t(index), entry, &bindings[0], C.uint32_t(len(bindings)), boolInt(kernel.parameterByteSize != 0), C.uint32_t(kernel.parameterBinding), C.uint32_t(kernel.parameterByteSize))
		C.free(unsafe.Pointer(entry))
		if status == 0 {
			C.tv_module_destroy(module)
			return s.fail(s.nativeError())
		}
	}
	id := s.nextModule
	s.nextModule++
	s.modules[id] = nativeModule{value: module, kernels: kernels}
	*output = C.uint32_t(id)
	return 0
}

//export tach_prepare
func tach_prepare(pointer unsafe.Pointer, moduleID C.uint32_t, kernels *C.uint32_t, count C.size_t) C.int32_t {
	s := resolve(pointer)
	if s == nil || kernels == nil || count == 0 || count > 1<<20 {
		return -1
	}
	s.Lock()
	defer s.Unlock()
	module, ok := s.modules[uint32(moduleID)]
	if !ok {
		return s.fail(fmt.Errorf("unknown native module %d", moduleID))
	}
	for _, index := range unsafe.Slice(kernels, int(count)) {
		if uint32(index) >= uint32(len(module.kernels)) {
			return s.fail(fmt.Errorf("invalid physical kernel %d", index))
		}
		if C.tv_module_prepare(module.value, index) == 0 {
			return s.fail(s.nativeError())
		}
	}
	return 0
}

//export tach_buffer
func tach_buffer(pointer unsafe.Pointer, size C.uint32_t, initial *C.uint8_t, output *C.uint32_t) C.int32_t {
	s := resolve(pointer)
	if s == nil || output == nil || initial == nil {
		return -1
	}
	s.Lock()
	defer s.Unlock()
	buffer := C.tv_buffer_create(s.context, size, initial)
	if buffer == nil {
		return s.fail(s.nativeError())
	}
	id := s.nextBuffer
	s.nextBuffer++
	s.buffers[id] = nativeBuffer{value: buffer, size: uint32(size)}
	*output = C.uint32_t(id)
	return 0
}

//export tach_write
func tach_write(pointer unsafe.Pointer, id C.uint32_t, bytes *C.uint8_t, length C.uint32_t) C.int32_t {
	s := resolve(pointer)
	if s == nil {
		return -1
	}
	return s.transfer(id, bytes, length, true)
}

//export tach_read
func tach_read(pointer unsafe.Pointer, id C.uint32_t, output *C.uint8_t, length C.uint32_t) C.int32_t {
	s := resolve(pointer)
	if s == nil {
		return -1
	}
	return s.transfer(id, output, length, false)
}

func (s *session) transfer(id C.uint32_t, bytes *C.uint8_t, length C.uint32_t, write bool) C.int32_t {
	s.Lock()
	defer s.Unlock()
	buffer, ok := s.buffers[uint32(id)]
	operation := "read"
	if write {
		operation = "write"
	}
	if !ok || bytes == nil || uint32(length) != buffer.size {
		return s.fail(fmt.Errorf("invalid buffer %s", operation))
	}
	var success C.int
	if write {
		success = C.tv_buffer_write(buffer.value, bytes, length)
	} else {
		success = C.tv_buffer_read(buffer.value, bytes, length)
	}
	if success == 0 {
		return s.fail(s.nativeError())
	}
	return 0
}

//export tach_destroy_buffer
func tach_destroy_buffer(pointer unsafe.Pointer, id C.uint32_t) {
	s := resolve(pointer)
	if s == nil {
		return
	}
	s.Lock()
	defer s.Unlock()
	if buffer, ok := s.buffers[uint32(id)]; ok {
		C.tv_buffer_destroy(buffer.value)
		delete(s.buffers, uint32(id))
	}
}

func (s *session) resource(module uint32, wire wireResource) (nativeBuffer, error) {
	if wire.kind == 0 {
		buffer, ok := s.buffers[wire.id]
		if !ok {
			return nativeBuffer{}, fmt.Errorf("unknown native buffer %d", wire.id)
		}
		return buffer, nil
	}
	buffer, ok := s.scratch[scratchKey{module: module, color: wire.id}]
	if !ok {
		return nativeBuffer{}, fmt.Errorf("unknown scratch color %d", wire.id)
	}
	return buffer, nil
}

func counts(commands []wireCommand, modules map[uint32]nativeModule) (sets, storage, uniforms, parameterBytes uint64, err error) {
	for _, command := range commands {
		module, ok := modules[command.module]
		if !ok {
			return 0, 0, 0, 0, fmt.Errorf("unknown native module %d", command.module)
		}
		for _, step := range command.steps {
			if step.barrier {
				continue
			}
			if int(step.kernel) >= len(module.kernels) || len(step.resources) != len(module.kernels[step.kernel].bindings) {
				return 0, 0, 0, 0, errors.New("dispatch does not match its native module")
			}
			sets += uint64(command.repeat)
			storage += uint64(command.repeat) * uint64(len(step.resources))
			if module.kernels[step.kernel].parameterByteSize != 0 {
				uniforms += uint64(command.repeat)
				parameterBytes += uint64(command.repeat) * uint64(len(step.parameters))
			}
		}
	}
	if sets > ^uint64(0)>>32 || storage > ^uint64(0)>>32 || uniforms > ^uint64(0)>>32 || parameterBytes > ^uint64(0)>>32 {
		return 0, 0, 0, 0, errors.New("native submission is too large")
	}
	return sets, storage, uniforms, parameterBytes, nil
}

//export tach_submit
func tach_submit(pointer unsafe.Pointer, batch *C.uint8_t, length C.size_t, output *C.uint64_t) C.int32_t {
	s := resolve(pointer)
	if s == nil || output == nil {
		return -1
	}
	commands, err := parseBatch(newReader(batch, length))
	if err != nil {
		s.Lock()
		defer s.Unlock()
		return s.fail(err)
	}
	s.Lock()
	defer s.Unlock()
	sets, storage, uniforms, parameterBytes, err := counts(commands, s.modules)
	if err != nil {
		return s.fail(err)
	}
	for _, command := range commands {
		for color, size := range command.scratch {
			key := scratchKey{module: command.module, color: color}
			if existing := s.scratch[key]; existing.size >= size {
				continue
			} else if existing.value != nil {
				C.tv_buffer_destroy(existing.value)
			}
			created := C.tv_buffer_create(s.context, C.uint32_t(size), nil)
			if created == nil {
				return s.fail(s.nativeError())
			}
			s.scratch[key] = nativeBuffer{value: created, size: size}
		}
	}
	var reuse *C.tv_submission
	if index := len(s.available) - 1; index >= 0 {
		reuse = s.available[index]
		s.available = s.available[:index]
	}
	submission := C.tv_submission_begin(s.context, reuse, C.uint32_t(sets), C.uint32_t(storage), C.uint32_t(uniforms), C.uint32_t(parameterBytes))
	if submission == nil {
		return s.fail(s.nativeError())
	}
	fail := func(err error) C.int32_t { C.tv_submission_destroy(submission); return s.fail(err) }
	for commandIndex, command := range commands {
		module := s.modules[command.module]
		for repeat := uint32(0); repeat < command.repeat; repeat++ {
			for _, step := range command.steps {
				if step.barrier {
					for _, resource := range step.resources {
						if _, err := s.resource(command.module, resource); err != nil {
							return fail(err)
						}
					}
					C.tv_submission_barrier(submission)
					continue
				}
				resources := make([]C.tv_resource, len(step.resources))
				for index, wire := range step.resources {
					buffer, err := s.resource(command.module, wire)
					if err != nil || wire.size > buffer.size {
						if err == nil {
							err = errors.New("native resource range exceeds its buffer")
						}
						return fail(err)
					}
					resources[index].binding = C.uint32_t(wire.binding)
					resources[index].buffer = buffer.value
					resources[index].size = C.uint32_t(wire.size)
				}
				var resourcePointer *C.tv_resource
				if len(resources) > 0 {
					resourcePointer = &resources[0]
				}
				var parameterPointer *C.uint8_t
				if len(step.parameters) > 0 {
					parameterPointer = (*C.uint8_t)(unsafe.Pointer(&step.parameters[0]))
				}
				if C.tv_submission_dispatch(submission, module.value, C.uint32_t(step.kernel), resourcePointer, C.uint32_t(len(resources)), parameterPointer,
					C.uint32_t(len(step.parameters)), C.uint32_t(step.groups[0]), C.uint32_t(step.groups[1]), C.uint32_t(step.groups[2])) == 0 {
					return fail(s.nativeError())
				}
			}
			if repeat+1 < command.repeat && len(command.repeatBarrier) > 0 {
				for _, resource := range command.repeatBarrier {
					if _, err := s.resource(command.module, resource); err != nil {
						return fail(err)
					}
				}
				C.tv_submission_barrier(submission)
			}
		}
		if commandIndex+1 < len(commands) {
			C.tv_submission_barrier(submission)
		}
	}
	if C.tv_submission_finish(submission) == 0 {
		return fail(s.nativeError())
	}
	id := s.nextSubmit
	s.nextSubmit++
	s.submissions[id] = submission
	*output = C.uint64_t(id)
	return 0
}

//export tach_wait
func tach_wait(pointer unsafe.Pointer, id C.uint64_t) C.int32_t {
	s := resolve(pointer)
	if s == nil {
		return -1
	}
	s.Lock()
	defer s.Unlock()
	submission, ok := s.submissions[uint64(id)]
	if !ok {
		return s.fail(fmt.Errorf("unknown submission %d", uint64(id)))
	}
	if C.tv_submission_wait(submission) == 0 {
		C.tv_submission_destroy(submission)
		delete(s.submissions, uint64(id))
		return s.fail(s.nativeError())
	}
	delete(s.submissions, uint64(id))
	s.available = append(s.available, submission)
	return 0
}

//export tach_error
func tach_error(pointer unsafe.Pointer, output *C.uint8_t, capacity C.size_t) C.int32_t {
	s := resolve(pointer)
	if s == nil {
		globalError.Lock()
		defer globalError.Unlock()
		return copyOutput(output, capacity, []byte(globalError.text))
	}
	s.Lock()
	defer s.Unlock()
	return copyOutput(output, capacity, []byte(s.err))
}

//export tach_close
func tach_close(pointer unsafe.Pointer) {
	s := resolve(pointer)
	if s == nil {
		return
	}
	s.Lock()
	for id, submission := range s.submissions {
		C.tv_submission_destroy(submission)
		delete(s.submissions, id)
	}
	for _, submission := range s.available {
		C.tv_submission_destroy(submission)
	}
	s.available = nil
	for id, buffer := range s.buffers {
		C.tv_buffer_destroy(buffer.value)
		delete(s.buffers, id)
	}
	for key, buffer := range s.scratch {
		C.tv_buffer_destroy(buffer.value)
		delete(s.scratch, key)
	}
	for id, module := range s.modules {
		C.tv_module_destroy(module.value)
		delete(s.modules, id)
	}
	C.tv_close(s.context)
	s.context = nil
	s.Unlock()
	s.handle.Delete()
}

func main() {}

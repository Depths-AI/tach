package spirvtest

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"unsafe"

	vk "github.com/goki/vulkan"
)

const validationLayer = "VK_LAYER_KHRONOS_validation"

type moduleMetadata struct {
	Schema   int                     `json:"schema"`
	Programs []publicProgramMetadata `json:"programs"`
	Targets  struct {
		SPIRV *targetMetadata `json:"spirv"`
	} `json:"targets"`
}

type resourceMetadata struct {
	Name            string     `json:"name"`
	ByteSize        uint32     `json:"byteSize"`
	MinimumByteSize uint32     `json:"minimumByteSize"`
	Runtime         bool       `json:"runtime"`
	RuntimeOffset   uint32     `json:"runtimeOffset"`
	RuntimeStride   uint32     `json:"runtimeStride"`
	Layout          hostLayout `json:"layout"`
}

type kernelMetadata struct {
	EntryPoint     string                  `json:"entryPoint"`
	WorkgroupSize  [3]uint32               `json:"workgroupSize"`
	Bindings       []bindingMetadata       `json:"bindings"`
	ParameterBlock *parameterBlockMetadata `json:"parameterBlock"`
}

type publicProgramMetadata struct {
	Name       string                    `json:"name"`
	Parameters []publicParameterMetadata `json:"parameters"`
	Resources  []resourceMetadata        `json:"resources"`
	Launch     *struct {
		Dimensions int `json:"dimensions"`
	} `json:"launch"`
}
type publicParameterMetadata struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"`
	Resource *int   `json:"resource"`
}
type bindingMetadata struct {
	Group           uint32 `json:"group"`
	Binding         uint32 `json:"binding"`
	MinimumByteSize uint32 `json:"minimumByteSize"`
}

type parameterBlockMetadata struct {
	Group    uint32                   `json:"group"`
	Binding  uint32                   `json:"binding"`
	ByteSize uint32                   `json:"byteSize"`
	Fields   []parameterFieldMetadata `json:"fields"`
}
type parameterFieldMetadata struct {
	ByteOffset uint32     `json:"byteOffset"`
	Layout     hostLayout `json:"layout"`
}
type hostLayout struct {
	Kind   string            `json:"kind"`
	Size   uint32            `json:"size"`
	Stride uint32            `json:"stride"`
	Count  uint32            `json:"count"`
	Fields []hostLayoutField `json:"fields"`
}
type hostLayoutField struct {
	Name   string     `json:"name"`
	Offset uint32     `json:"offset"`
	Type   hostLayout `json:"type"`
}
type targetMetadata struct {
	Kernels  []kernelMetadata      `json:"kernels"`
	Programs []programPlanMetadata `json:"programs"`
}
type programPlanMetadata struct {
	Program       int                 `json:"program"`
	Transients    []transientMetadata `json:"transients"`
	Steps         []stepMetadata      `json:"steps"`
	RepeatBarrier *stepMetadata       `json:"repeatBarrier"`
	Repeat        string              `json:"repeat"`
}
type transientMetadata struct {
	Stride          uint32          `json:"stride"`
	MinimumByteSize uint32          `json:"minimumByteSize"`
	Length          shapeExpression `json:"length"`
	Color           int             `json:"color"`
}
type stepMetadata struct {
	Kind       string                   `json:"kind"`
	Kernel     int                      `json:"kernel"`
	Domain     []shapeExpression        `json:"domain"`
	Resources  []resourceSourceMetadata `json:"resources"`
	Parameters []valueSourceMetadata    `json:"parameters"`
}
type resourceSourceMetadata struct {
	Binding  uint32 `json:"binding"`
	Kind     string `json:"kind"`
	Resource int    `json:"resource"`
}
type shapeExpression struct {
	Op        string           `json:"op"`
	Value     uint32           `json:"value"`
	Parameter int              `json:"parameter"`
	Resource  int              `json:"resource"`
	Path      []string         `json:"path"`
	Axis      uint8            `json:"axis"`
	Left      *shapeExpression `json:"left"`
	Right     *shapeExpression `json:"right"`
}
type valueSourceMetadata struct {
	Kind       string           `json:"kind"`
	Parameter  int              `json:"parameter"`
	Path       []string         `json:"path"`
	Value      any              `json:"value"`
	Expression *shapeExpression `json:"expression"`
}

type adapter struct {
	Name       string
	Type       string
	Mode       string
	APIVersion string
	VendorID   uint32
	DeviceID   uint32
}

type validationMessage struct {
	flags   vk.DebugReportFlags
	message string
}

type vulkanHarness struct {
	instance          vk.Instance
	physical          vk.PhysicalDevice
	device            vk.Device
	queue             vk.Queue
	queueFamily       uint32
	callback          vk.DebugReportCallback
	callbackInstalled bool
	validationEnabled bool
	adapter           adapter

	validationMu sync.Mutex
	validation   []validationMessage
}

func openVulkan() (*vulkanHarness, error) {
	if err := vk.SetDefaultGetInstanceProcAddr(); err != nil {
		return nil, fmt.Errorf("load Vulkan: %w", err)
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("initialize Vulkan: %w", err)
	}

	layerAvailable, err := instanceLayerAvailable(validationLayer)
	if err != nil {
		return nil, err
	}
	debugAvailable, err := instanceExtensionAvailable(vk.ExtDebugReportExtensionName)
	if err != nil {
		return nil, err
	}

	layers := []string(nil)
	extensions := []string(nil)
	validationEnabled := layerAvailable && debugAvailable
	if validationEnabled {
		layers = []string{validationLayer + "\x00"}
		extensions = []string{vk.ExtDebugReportExtensionName + "\x00"}
	}
	application := vk.ApplicationInfo{
		SType:            vk.StructureTypeApplicationInfo,
		PApplicationName: "Tach SPIR-V test\x00",
		PEngineName:      "Tach\x00",
		ApiVersion:       vk.MakeVersion(1, 1, 0),
	}
	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &application,
		EnabledLayerCount:       uint32(len(layers)),
		PpEnabledLayerNames:     layers,
		EnabledExtensionCount:   uint32(len(extensions)),
		PpEnabledExtensionNames: extensions,
	}
	h := &vulkanHarness{validationEnabled: validationEnabled}
	var instance vk.Instance
	if err := result("create instance", vk.CreateInstance(&createInfo, nil, &instance)); err != nil {
		return nil, err
	}
	h.instance = instance
	if err := vk.InitInstance(h.instance); err != nil {
		vk.DestroyInstance(h.instance, nil)
		return nil, fmt.Errorf("initialize instance functions: %w", err)
	}
	if validationEnabled {
		callbackInfo := vk.DebugReportCallbackCreateInfo{
			SType: vk.StructureTypeDebugReportCallbackCreateInfo,
			Flags: vk.DebugReportFlags(vk.DebugReportErrorBit |
				vk.DebugReportWarningBit | vk.DebugReportPerformanceWarningBit),
			PfnCallback: h.captureValidation,
		}
		var callback vk.DebugReportCallback
		if err := result("create validation callback", vk.CreateDebugReportCallback(
			h.instance, &callbackInfo, nil, &callback,
		)); err != nil {
			vk.DestroyInstance(h.instance, nil)
			return nil, err
		}
		h.callback = callback
		h.callbackInstalled = true
	}

	physical, family, properties, err := selectPhysicalDevice(h.instance)
	if err != nil {
		h.close()
		return nil, err
	}
	h.physical = physical
	h.queueFamily = family
	h.adapter = describeAdapter(properties)

	priority := []float32{1}
	queueInfo := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: family,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	features := vk.PhysicalDeviceFeatures{RobustBufferAccess: vk.True}
	deviceInfo := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{queueInfo},
		PEnabledFeatures:     []vk.PhysicalDeviceFeatures{features},
	}
	var device vk.Device
	if err := result("create compute device", vk.CreateDevice(physical, &deviceInfo, nil, &device)); err != nil {
		h.close()
		return nil, err
	}
	h.device = device
	var queue vk.Queue
	vk.GetDeviceQueue(h.device, family, 0, &queue)
	h.queue = queue
	if err := validationError(h.validationSince(0)); err != nil {
		h.close()
		return nil, err
	}
	return h, nil
}

func (h *vulkanHarness) close() {
	if h.device != nil {
		_ = vk.DeviceWaitIdle(h.device)
		vk.DestroyDevice(h.device, nil)
		h.device = nil
	}
	if h.callbackInstalled {
		vk.DestroyDebugReportCallback(h.instance, h.callback, nil)
		h.callbackInstalled = false
	}
	if h.instance != nil {
		vk.DestroyInstance(h.instance, nil)
		h.instance = nil
	}
}

func (h *vulkanHarness) captureValidation(
	flags vk.DebugReportFlags,
	_ vk.DebugReportObjectType,
	_ uint64,
	_ uint64,
	_ int32,
	layer string,
	message string,
	_ unsafe.Pointer,
) vk.Bool32 {
	h.validationMu.Lock()
	h.validation = append(h.validation, validationMessage{
		flags:   flags,
		message: strings.TrimSpace(layer + ": " + message),
	})
	h.validationMu.Unlock()
	return vk.False
}

func (h *vulkanHarness) validationMark() int {
	h.validationMu.Lock()
	defer h.validationMu.Unlock()
	return len(h.validation)
}

func (h *vulkanHarness) validationSince(mark int) []validationMessage {
	h.validationMu.Lock()
	defer h.validationMu.Unlock()
	return append([]validationMessage(nil), h.validation[mark:]...)
}

func instanceLayerAvailable(name string) (bool, error) {
	var count uint32
	if err := result("enumerate instance layers", vk.EnumerateInstanceLayerProperties(&count, nil)); err != nil {
		return false, err
	}
	properties := make([]vk.LayerProperties, count)
	if count > 0 {
		if err := result("enumerate instance layers", vk.EnumerateInstanceLayerProperties(&count, properties)); err != nil {
			return false, err
		}
	}
	for i := range properties {
		properties[i].Deref()
		if vk.ToString(properties[i].LayerName[:]) == name {
			return true, nil
		}
	}
	return false, nil
}

func instanceExtensionAvailable(name string) (bool, error) {
	var count uint32
	if err := result("enumerate instance extensions", vk.EnumerateInstanceExtensionProperties("", &count, nil)); err != nil {
		return false, err
	}
	properties := make([]vk.ExtensionProperties, count)
	if count > 0 {
		if err := result("enumerate instance extensions", vk.EnumerateInstanceExtensionProperties("", &count, properties)); err != nil {
			return false, err
		}
	}
	for i := range properties {
		properties[i].Deref()
		if vk.ToString(properties[i].ExtensionName[:]) == name {
			return true, nil
		}
	}
	return false, nil
}

func selectPhysicalDevice(instance vk.Instance) (
	vk.PhysicalDevice,
	uint32,
	vk.PhysicalDeviceProperties,
	error,
) {
	var count uint32
	if err := result("enumerate physical devices", vk.EnumeratePhysicalDevices(instance, &count, nil)); err != nil {
		return nil, 0, vk.PhysicalDeviceProperties{}, err
	}
	if count == 0 {
		return nil, 0, vk.PhysicalDeviceProperties{}, errors.New("no Vulkan physical device")
	}
	devices := make([]vk.PhysicalDevice, count)
	if err := result("enumerate physical devices", vk.EnumeratePhysicalDevices(instance, &count, devices)); err != nil {
		return nil, 0, vk.PhysicalDeviceProperties{}, err
	}

	type candidate struct {
		device     vk.PhysicalDevice
		family     uint32
		properties vk.PhysicalDeviceProperties
		rank       int
	}
	var candidates []candidate
	for _, device := range devices {
		var properties vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(device, &properties)
		properties.Deref()
		if vk.Version(properties.ApiVersion).Major() < 1 ||
			(vk.Version(properties.ApiVersion).Major() == 1 && vk.Version(properties.ApiVersion).Minor() < 1) {
			continue
		}
		var features vk.PhysicalDeviceFeatures
		vk.GetPhysicalDeviceFeatures(device, &features)
		features.Deref()
		if !features.RobustBufferAccess.B() {
			continue
		}
		family, ok := computeQueueFamily(device)
		if !ok {
			continue
		}
		candidates = append(candidates, candidate{
			device: device, family: family, properties: properties, rank: deviceRank(properties.DeviceType),
		})
	}
	if len(candidates) == 0 {
		return nil, 0, vk.PhysicalDeviceProperties{},
			errors.New("no Vulkan 1.1 compute device with robust buffer access")
	}
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].rank < candidates[j].rank })
	selected := candidates[0]
	return selected.device, selected.family, selected.properties, nil
}

func computeQueueFamily(device vk.PhysicalDevice) (uint32, bool) {
	var count uint32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, nil)
	properties := make([]vk.QueueFamilyProperties, count)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, properties)
	for i := range properties {
		properties[i].Deref()
		if properties[i].QueueCount > 0 && properties[i].QueueFlags&vk.QueueFlags(vk.QueueComputeBit) != 0 {
			return uint32(i), true
		}
	}
	return 0, false
}

func deviceRank(deviceType vk.PhysicalDeviceType) int {
	switch deviceType {
	case vk.PhysicalDeviceTypeDiscreteGpu:
		return 0
	case vk.PhysicalDeviceTypeIntegratedGpu:
		return 1
	case vk.PhysicalDeviceTypeVirtualGpu:
		return 2
	case vk.PhysicalDeviceTypeCpu:
		return 4
	default:
		return 3
	}
}

func describeAdapter(properties vk.PhysicalDeviceProperties) adapter {
	name := vk.ToString(properties.DeviceName[:])
	typeName := physicalDeviceType(properties.DeviceType)
	software := properties.DeviceType == vk.PhysicalDeviceTypeCpu ||
		containsSoftwareIdentity(name)
	mode := "hardware-accelerated"
	if software {
		mode = "software-emulated"
	}
	return adapter{
		Name:       name,
		Type:       typeName,
		Mode:       mode,
		APIVersion: vk.Version(properties.ApiVersion).String(),
		VendorID:   properties.VendorID,
		DeviceID:   properties.DeviceID,
	}
}

func physicalDeviceType(deviceType vk.PhysicalDeviceType) string {
	switch deviceType {
	case vk.PhysicalDeviceTypeDiscreteGpu:
		return "discrete GPU"
	case vk.PhysicalDeviceTypeIntegratedGpu:
		return "integrated GPU"
	case vk.PhysicalDeviceTypeVirtualGpu:
		return "virtual GPU"
	case vk.PhysicalDeviceTypeCpu:
		return "CPU"
	default:
		return "other"
	}
}

func containsSoftwareIdentity(name string) bool {
	name = strings.ToLower(name)
	for _, marker := range []string{"llvmpipe", "lavapipe", "swiftshader", "softpipe", "software", "warp", "basic render"} {
		if strings.Contains(name, marker) {
			return true
		}
	}
	return false
}

func validationError(messages []validationMessage) error {
	if len(messages) == 0 {
		return nil
	}
	lines := make([]string, len(messages))
	for i, message := range messages {
		severity := "warning"
		if message.flags&vk.DebugReportFlags(vk.DebugReportErrorBit) != 0 {
			severity = "error"
		} else if message.flags&vk.DebugReportFlags(vk.DebugReportPerformanceWarningBit) != 0 {
			severity = "performance"
		}
		lines[i] = fmt.Sprintf("[%s] %s", severity, message.message)
	}
	return fmt.Errorf("Vulkan validation: %s", strings.Join(lines, "\n"))
}

func (h *vulkanHarness) createBuffer(kind string, data []byte) (vk.Buffer, vk.DeviceMemory, vk.DeviceSize, bool, error) {
	usage := vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit)
	if kind == "uniform" {
		usage = vk.BufferUsageFlags(vk.BufferUsageUniformBufferBit)
	} else if kind != "storage" {
		return vk.NullBuffer, vk.NullDeviceMemory, 0, false, fmt.Errorf("unknown resource kind %q", kind)
	}
	info := vk.BufferCreateInfo{SType: vk.StructureTypeBufferCreateInfo, Size: vk.DeviceSize(len(data)), Usage: usage, SharingMode: vk.SharingModeExclusive}
	var buffer vk.Buffer
	if err := result("create buffer", vk.CreateBuffer(h.device, &info, nil, &buffer)); err != nil {
		return vk.NullBuffer, vk.NullDeviceMemory, 0, false, err
	}
	var requirements vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(h.device, buffer, &requirements)
	requirements.Deref()
	memoryType, coherent := vk.FindMemoryTypeIndex(h.physical, requirements.MemoryTypeBits, vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit)
	if !coherent {
		var ok bool
		memoryType, ok = vk.FindMemoryTypeIndex(h.physical, requirements.MemoryTypeBits, vk.MemoryPropertyHostVisibleBit)
		if !ok {
			vk.DestroyBuffer(h.device, buffer, nil)
			return vk.NullBuffer, vk.NullDeviceMemory, 0, false, errors.New("no host-visible Vulkan memory type")
		}
	}
	allocateInfo := vk.MemoryAllocateInfo{SType: vk.StructureTypeMemoryAllocateInfo, AllocationSize: requirements.Size, MemoryTypeIndex: memoryType}
	var memory vk.DeviceMemory
	if err := result("allocate buffer memory", vk.AllocateMemory(h.device, &allocateInfo, nil, &memory)); err != nil {
		vk.DestroyBuffer(h.device, buffer, nil)
		return vk.NullBuffer, vk.NullDeviceMemory, 0, false, err
	}
	if err := result("bind buffer memory", vk.BindBufferMemory(h.device, buffer, memory, 0)); err != nil {
		vk.FreeMemory(h.device, memory, nil)
		vk.DestroyBuffer(h.device, buffer, nil)
		return vk.NullBuffer, vk.NullDeviceMemory, 0, false, err
	}
	var mapped unsafe.Pointer
	if err := result("map input buffer", vk.MapMemory(h.device, memory, 0, requirements.Size, 0, &mapped)); err != nil {
		vk.DestroyBuffer(h.device, buffer, nil)
		vk.FreeMemory(h.device, memory, nil)
		return vk.NullBuffer, vk.NullDeviceMemory, 0, false, err
	}
	copy(unsafe.Slice((*byte)(mapped), len(data)), data)
	if !coherent {
		rangeInfo := vk.MappedMemoryRange{SType: vk.StructureTypeMappedMemoryRange, Memory: memory, Size: vk.DeviceSize(vk.WholeSize)}
		if err := result("flush input buffer", vk.FlushMappedMemoryRanges(h.device, 1, []vk.MappedMemoryRange{rangeInfo})); err != nil {
			vk.UnmapMemory(h.device, memory)
			vk.DestroyBuffer(h.device, buffer, nil)
			vk.FreeMemory(h.device, memory, nil)
			return vk.NullBuffer, vk.NullDeviceMemory, 0, false, err
		}
	}
	vk.UnmapMemory(h.device, memory)
	return buffer, memory, requirements.Size, coherent, nil
}

func result(operation string, value vk.Result) error {
	if err := vk.Error(value); err != nil {
		return fmt.Errorf("%s: %w", operation, err)
	}
	return nil
}

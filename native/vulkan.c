//go:build tachvulkan

#include "vulkan.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define INSTANCE_FUNCTIONS(X) \
  X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
  X(GetPhysicalDeviceFeatures2) X(GetPhysicalDeviceQueueFamilyProperties) \
  X(GetPhysicalDeviceMemoryProperties) X(CreateDevice) X(GetDeviceProcAddr)

#define DEVICE_FUNCTIONS(X) \
  X(DestroyDevice) X(GetDeviceQueue) X(DeviceWaitIdle) \
  X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) \
  X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
  X(FlushMappedMemoryRanges) X(InvalidateMappedMemoryRanges) \
  X(CreateShaderModule) X(DestroyShaderModule) \
  X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
  X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
  X(CreateDescriptorPool) X(DestroyDescriptorPool) X(ResetDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
  X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(FreeCommandBuffers) X(ResetCommandBuffer) \
  X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdCopyBuffer) X(CmdPipelineBarrier2) \
  X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdDispatch) \
  X(CreateFence) X(DestroyFence) X(ResetFences) X(WaitForFences) X(QueueSubmit2)

typedef struct {
  char *entry;
  tv_binding *bindings;
  VkDescriptorSetLayout descriptors;
  VkPipelineLayout layout;
  VkPipeline pipeline;
  uint32_t binding_count;
  int has_parameters;
  uint32_t parameter_binding;
  uint32_t parameter_size;
} tv_kernel;

struct tv_context {
#ifdef _WIN32
  HMODULE loader;
#else
  void *loader;
#endif
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  VkInstance instance;
  VkPhysicalDevice physical;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memory;
  uint32_t loader_version;
  VkCommandPool command_pool;
  char error[512];
#define FIELD(name) PFN_vk##name name;
  INSTANCE_FUNCTIONS(FIELD)
  DEVICE_FUNCTIONS(FIELD)
#undef FIELD
};

struct tv_module {
  tv_context *context;
  VkShaderModule shader;
  tv_kernel *kernels;
  uint32_t kernel_count;
};

struct tv_buffer {
  tv_context *context;
  VkBuffer buffer;
  VkDeviceMemory memory;
  void *mapped;
  VkDeviceSize allocation_size;
  uint32_t size;
  VkMemoryPropertyFlags properties;
};

struct tv_submission {
  tv_context *context;
  VkDescriptorPool descriptors;
  VkCommandBuffer commands;
  VkFence fence;
  tv_buffer *parameters;
  uint32_t set_capacity;
  uint32_t storage_capacity;
  uint32_t uniform_capacity;
  uint32_t parameter_capacity;
  uint32_t parameter_offset;
  int submitted;
};

static char global_error[512];

static int set_error(tv_context *context, const char *format, ...) {
  char *destination = context ? context->error : global_error;
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(destination, 512, format, arguments);
  va_end(arguments);
  return 0;
}

static int result(tv_context *context, const char *operation, VkResult value) {
  return value == VK_SUCCESS ? 1 : set_error(context, "%s failed with VkResult %d", operation, value);
}

static char *copy_string(const char *source) {
  size_t length = strlen(source) + 1;
  char *copy = (char *)malloc(length);
  if (copy) memcpy(copy, source, length);
  return copy;
}

static uint32_t version_at_least(uint32_t value, uint32_t required) {
  return VK_API_VERSION_MAJOR(value) > VK_API_VERSION_MAJOR(required) ||
    (VK_API_VERSION_MAJOR(value) == VK_API_VERSION_MAJOR(required) &&
     VK_API_VERSION_MINOR(value) >= VK_API_VERSION_MINOR(required));
}

static void unload(tv_context *context) {
  if (!context->loader) return;
#ifdef _WIN32
  FreeLibrary(context->loader);
#else
  dlclose(context->loader);
#endif
  context->loader = 0;
}

static int load_global(tv_context *context) {
#ifdef _WIN32
  context->loader = LoadLibraryA("vulkan-1.dll");
  if (context->loader) context->get_instance_proc_addr = (PFN_vkGetInstanceProcAddr)GetProcAddress(context->loader, "vkGetInstanceProcAddr");
#else
  context->loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (context->loader) context->get_instance_proc_addr = (PFN_vkGetInstanceProcAddr)dlsym(context->loader, "vkGetInstanceProcAddr");
#endif
  return context->get_instance_proc_addr != NULL || set_error(context, "Vulkan loader or vkGetInstanceProcAddr is unavailable");
}

static int load_instance(tv_context *context) {
#define LOAD(name) context->name = (PFN_vk##name)context->get_instance_proc_addr(context->instance, "vk" #name); if (!context->name) return set_error(context, "required Vulkan 1.3 function vk%s is unavailable", #name);
  INSTANCE_FUNCTIONS(LOAD)
#undef LOAD
  return 1;
}

static int load_device(tv_context *context) {
#define LOAD(name) context->name = (PFN_vk##name)context->GetDeviceProcAddr(context->device, "vk" #name); if (!context->name) return set_error(context, "required Vulkan 1.3 function vk%s is unavailable", #name);
  DEVICE_FUNCTIONS(LOAD)
#undef LOAD
  return 1;
}

static int queue_family(tv_context *context, VkPhysicalDevice physical, uint32_t *selected) {
  uint32_t count = 0;
  context->GetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
  VkQueueFamilyProperties *properties = count ? (VkQueueFamilyProperties *)calloc(count, sizeof(*properties)) : NULL;
  if (!properties) return 0;
  context->GetPhysicalDeviceQueueFamilyProperties(physical, &count, properties);
  int found = 0;
  for (uint32_t index = 0; index < count; ++index) {
    if (properties[index].queueCount && (properties[index].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      if (!found || !(properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) *selected = index;
      found = 1;
      if (!(properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
    }
  }
  free(properties);
  return found;
}

static int device_rank(VkPhysicalDeviceType type, int high_performance) {
  if (high_performance) {
    if (type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 0;
    if (type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 1;
  } else {
    if (type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 0;
    if (type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 1;
  }
  if (type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) return 2;
  if (type == VK_PHYSICAL_DEVICE_TYPE_CPU) return 4;
  return 3;
}

static int select_device(tv_context *context, int high_performance) {
  uint32_t count = 0;
  if (!result(context, "vkEnumeratePhysicalDevices", context->EnumeratePhysicalDevices(context->instance, &count, NULL)) || !count)
    return count ? 0 : set_error(context, "no Vulkan physical device is available");
  VkPhysicalDevice *devices = (VkPhysicalDevice *)calloc(count, sizeof(*devices));
  if (!devices) return set_error(context, "out of memory while enumerating Vulkan devices");
  if (!result(context, "vkEnumeratePhysicalDevices", context->EnumeratePhysicalDevices(context->instance, &count, devices))) { free(devices); return 0; }
  int best_rank = 99;
  for (uint32_t index = 0; index < count; ++index) {
    VkPhysicalDeviceProperties properties;
    context->GetPhysicalDeviceProperties(devices[index], &properties);
    if (!version_at_least(properties.apiVersion, VK_API_VERSION_1_3)) continue;
    VkPhysicalDeviceVulkan13Features features13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceFeatures2 features = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features13 };
    context->GetPhysicalDeviceFeatures2(devices[index], &features);
    if (!features.features.robustBufferAccess || !features13.synchronization2 || !features13.shaderZeroInitializeWorkgroupMemory) continue;
    uint32_t family;
    if (!queue_family(context, devices[index], &family)) continue;
    int rank = device_rank(properties.deviceType, high_performance);
    if (rank < best_rank) {
      best_rank = rank; context->physical = devices[index]; context->queue_family = family; context->properties = properties;
    }
  }
  free(devices);
  if (!context->physical) return set_error(context, "no Vulkan 1.3 compute device supports robustBufferAccess, synchronization2, and shaderZeroInitializeWorkgroupMemory");
  context->GetPhysicalDeviceMemoryProperties(context->physical, &context->memory);
  return 1;
}

tv_context *tv_open(int high_performance) {
  tv_context *context = (tv_context *)calloc(1, sizeof(*context));
  if (!context) { set_error(NULL, "out of memory while opening Vulkan"); return NULL; }
  if (!load_global(context)) goto fail;
  PFN_vkEnumerateInstanceVersion enumerate_version = (PFN_vkEnumerateInstanceVersion)context->get_instance_proc_addr(NULL, "vkEnumerateInstanceVersion");
  PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)context->get_instance_proc_addr(NULL, "vkCreateInstance");
  if (!enumerate_version || !create_instance || !result(context, "vkEnumerateInstanceVersion", enumerate_version(&context->loader_version))) goto fail;
  if (!version_at_least(context->loader_version, VK_API_VERSION_1_3)) { set_error(context, "Vulkan loader %u.%u is below the required 1.3 floor", VK_API_VERSION_MAJOR(context->loader_version), VK_API_VERSION_MINOR(context->loader_version)); goto fail; }
  VkApplicationInfo application = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "Tach", .applicationVersion = 1, .pEngineName = "Tach", .engineVersion = 1, .apiVersion = VK_API_VERSION_1_3 };
  VkInstanceCreateInfo instance_info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &application };
  if (!result(context, "vkCreateInstance", create_instance(&instance_info, NULL, &context->instance)) || !load_instance(context) || !select_device(context, high_performance)) goto fail;
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = context->queue_family, .queueCount = 1, .pQueuePriorities = &priority };
  VkPhysicalDeviceVulkan13Features features13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .synchronization2 = VK_TRUE, .shaderZeroInitializeWorkgroupMemory = VK_TRUE };
  VkPhysicalDeviceFeatures2 features = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features13, .features.robustBufferAccess = VK_TRUE };
  VkDeviceCreateInfo device_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features, .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue };
  if (!result(context, "vkCreateDevice", context->CreateDevice(context->physical, &device_info, NULL, &context->device)) || !load_device(context)) goto fail;
  context->GetDeviceQueue(context->device, context->queue_family, 0, &context->queue);
  VkCommandPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = context->queue_family };
  if (!result(context, "vkCreateCommandPool", context->CreateCommandPool(context->device, &pool, NULL, &context->command_pool))) goto fail;
  return context;
fail:
  strncpy(global_error, context->error, sizeof(global_error) - 1);
  tv_close(context);
  return NULL;
}

const char *tv_global_error(void) { return global_error; }
const char *tv_error(tv_context *context) { return context ? context->error : global_error; }
const char *tv_device_name(tv_context *context) { return context->properties.deviceName; }
uint32_t tv_vendor_id(tv_context *context) { return context->properties.vendorID; }
uint32_t tv_device_type(tv_context *context) { return context->properties.deviceType; }

void tv_close(tv_context *context) {
  if (!context) return;
  if (context->device) {
    if (context->DeviceWaitIdle) context->DeviceWaitIdle(context->device);
    if (context->command_pool && context->DestroyCommandPool) context->DestroyCommandPool(context->device, context->command_pool, NULL);
    if (context->DestroyDevice) context->DestroyDevice(context->device, NULL);
  }
  if (context->instance && context->DestroyInstance) context->DestroyInstance(context->instance, NULL);
  unload(context);
  free(context);
}

static int memory_type(tv_context *context, uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, uint32_t *selected) {
  int fallback = -1;
  for (uint32_t index = 0; index < context->memory.memoryTypeCount; ++index) {
    if (!(bits & (1u << index)) || (context->memory.memoryTypes[index].propertyFlags & required) != required) continue;
    if ((context->memory.memoryTypes[index].propertyFlags & preferred) == preferred) { *selected = index; return 1; }
    if (fallback < 0) fallback = (int)index;
  }
  if (fallback >= 0) { *selected = (uint32_t)fallback; return 1; }
  return set_error(context, "no Vulkan memory type satisfies required flags 0x%x", required);
}

static tv_buffer *make_buffer(tv_context *context, uint32_t size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
  tv_buffer *buffer = (tv_buffer *)calloc(1, sizeof(*buffer));
  if (!buffer) { set_error(context, "out of memory while creating a buffer"); return NULL; }
  buffer->context = context; buffer->size = size;
  VkBufferCreateInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
  if (!result(context, "vkCreateBuffer", context->CreateBuffer(context->device, &info, NULL, &buffer->buffer))) goto fail;
  VkMemoryRequirements requirements;
  context->GetBufferMemoryRequirements(context->device, buffer->buffer, &requirements);
  uint32_t type;
  if (!memory_type(context, requirements.memoryTypeBits, required, preferred, &type)) goto fail;
  buffer->allocation_size = requirements.size;
  buffer->properties = context->memory.memoryTypes[type].propertyFlags;
  VkMemoryAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size, .memoryTypeIndex = type };
  if (!result(context, "vkAllocateMemory", context->AllocateMemory(context->device, &allocation, NULL, &buffer->memory)) ||
      !result(context, "vkBindBufferMemory", context->BindBufferMemory(context->device, buffer->buffer, buffer->memory, 0))) goto fail;
  return buffer;
fail:
  if (buffer->memory) context->FreeMemory(context->device, buffer->memory, NULL);
  if (buffer->buffer) context->DestroyBuffer(context->device, buffer->buffer, NULL);
  free(buffer);
  return NULL;
}

static void free_buffer(tv_buffer *buffer) {
  if (!buffer) return;
  tv_context *context = buffer->context;
  if (buffer->mapped) context->UnmapMemory(context->device, buffer->memory);
  if (buffer->buffer) context->DestroyBuffer(context->device, buffer->buffer, NULL);
  if (buffer->memory) context->FreeMemory(context->device, buffer->memory, NULL);
  free(buffer);
}

static int host_copy(tv_buffer *buffer, uint8_t *bytes, uint32_t length, uint32_t offset, int write) {
  tv_context *context = buffer->context;
  if ((uint64_t)offset + length > buffer->size) return set_error(context, "host copy exceeds its buffer");
  void *mapped = NULL;
  if (!result(context, "vkMapMemory", context->MapMemory(context->device, buffer->memory, 0, buffer->allocation_size, 0, &mapped))) return 0;
  if (!write && !(buffer->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = buffer->memory, .offset = 0, .size = VK_WHOLE_SIZE };
    if (!result(context, "vkInvalidateMappedMemoryRanges", context->InvalidateMappedMemoryRanges(context->device, 1, &range))) { context->UnmapMemory(context->device, buffer->memory); return 0; }
  }
  if (write) memcpy((uint8_t *)mapped + offset, bytes, length); else memcpy(bytes, (uint8_t *)mapped + offset, length);
  if (write && !(buffer->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = buffer->memory, .offset = 0, .size = VK_WHOLE_SIZE };
    if (!result(context, "vkFlushMappedMemoryRanges", context->FlushMappedMemoryRanges(context->device, 1, &range))) { context->UnmapMemory(context->device, buffer->memory); return 0; }
  }
  context->UnmapMemory(context->device, buffer->memory);
  return 1;
}

static int allocate_commands(tv_context *context, VkCommandBuffer *commands) {
  VkCommandBufferAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = context->command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
  if (!result(context, "vkAllocateCommandBuffers", context->AllocateCommandBuffers(context->device, &allocation, commands))) return 0;
  VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
  if (!result(context, "vkBeginCommandBuffer", context->BeginCommandBuffer(*commands, &begin))) {
    context->FreeCommandBuffers(context->device, context->command_pool, 1, commands); return 0;
  }
  return 1;
}

static int queue_commands(tv_context *context, VkCommandBuffer commands, VkFence fence) {
  VkCommandBufferSubmitInfo command = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = commands };
  VkSubmitInfo2 submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &command };
  return result(context, "vkQueueSubmit2", context->QueueSubmit2(context->queue, 1, &submit, fence));
}

static void transfer_barrier(tv_context *context, VkCommandBuffer commands,
                             VkPipelineStageFlags2 source_stage, VkAccessFlags2 source_access,
                             VkPipelineStageFlags2 destination_stage, VkAccessFlags2 destination_access) {
  VkMemoryBarrier2 memory = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = source_stage, .srcAccessMask = source_access,
    .dstStageMask = destination_stage, .dstAccessMask = destination_access };
  VkDependencyInfo dependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory };
  context->CmdPipelineBarrier2(commands, &dependency);
}

static int immediate_transfer(tv_buffer *device_buffer, tv_buffer *staging, int upload) {
  tv_context *context = device_buffer->context;
  if (!result(context, "vkDeviceWaitIdle", context->DeviceWaitIdle(context->device))) return 0;
  VkCommandBuffer commands;
  if (!allocate_commands(context, &commands)) return 0;
  VkBufferCopy copy = { .size = device_buffer->size };
  if (upload) {
    context->CmdCopyBuffer(commands, staging->buffer, device_buffer->buffer, 1, &copy);
    transfer_barrier(context, commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  } else {
    transfer_barrier(context, commands, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    context->CmdCopyBuffer(commands, device_buffer->buffer, staging->buffer, 1, &copy);
  }
  int ok = result(context, "vkEndCommandBuffer", context->EndCommandBuffer(commands));
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  if (ok) ok = result(context, "vkCreateFence", context->CreateFence(context->device, &fence_info, NULL, &fence));
  if (ok) ok = queue_commands(context, commands, fence);
  if (ok) ok = result(context, "vkWaitForFences", context->WaitForFences(context->device, 1, &fence, VK_TRUE, UINT64_MAX));
  if (fence) context->DestroyFence(context->device, fence, NULL);
  context->FreeCommandBuffers(context->device, context->command_pool, 1, &commands);
  return ok;
}

tv_buffer *tv_buffer_create(tv_context *context, uint32_t size, const uint8_t *initial) {
  if (!context || !size || size > context->properties.limits.maxStorageBufferRange) { if (context) set_error(context, "invalid storage buffer size %u", size); return NULL; }
  tv_buffer *buffer = make_buffer(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
  if (!buffer) return NULL;
  if (initial && !tv_buffer_write(buffer, initial, size)) { free_buffer(buffer); return NULL; }
  return buffer;
}

int tv_buffer_write(tv_buffer *buffer, const uint8_t *bytes, uint32_t length) {
  if (!buffer || !bytes || length != buffer->size) return buffer ? set_error(buffer->context, "buffer write length %u does not equal %u", length, buffer->size) : 0;
  tv_buffer *staging = make_buffer(buffer->context, length, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (!staging) return 0;
  int ok = host_copy(staging, (uint8_t *)bytes, length, 0, 1) && immediate_transfer(buffer, staging, 1);
  free_buffer(staging);
  return ok;
}

int tv_buffer_read(tv_buffer *buffer, uint8_t *output, uint32_t length) {
  if (!buffer || !output || length != buffer->size) return buffer ? set_error(buffer->context, "buffer read length %u does not equal %u", length, buffer->size) : 0;
  tv_buffer *staging = make_buffer(buffer->context, length, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (!staging) return 0;
  int ok = immediate_transfer(buffer, staging, 0) && host_copy(staging, output, length, 0, 0);
  free_buffer(staging);
  return ok;
}

void tv_buffer_destroy(tv_buffer *buffer) {
  if (!buffer) return;
  buffer->context->DeviceWaitIdle(buffer->context->device);
  free_buffer(buffer);
}

tv_module *tv_module_create(tv_context *context, const uint8_t *spirv, size_t length, uint32_t kernel_count) {
  uint32_t magic = 0, version = 0;
  if (!context || !spirv || length < 20 || length % 4 || !kernel_count) return context ? (set_error(context, "invalid SPIR-V module"), NULL) : NULL;
  memcpy(&magic, spirv, 4); memcpy(&version, spirv + 4, 4);
  if (magic != 0x07230203 || version != 0x00010600) { set_error(context, "SPIR-V module must use version 1.6"); return NULL; }
  tv_module *module = (tv_module *)calloc(1, sizeof(*module));
  if (!module) { set_error(context, "out of memory while creating a module"); return NULL; }
  module->context = context; module->kernel_count = kernel_count;
  module->kernels = (tv_kernel *)calloc(kernel_count, sizeof(*module->kernels));
  if (!module->kernels) { free(module); set_error(context, "out of memory while creating kernels"); return NULL; }
  VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = length, .pCode = (const uint32_t *)spirv };
  if (!result(context, "vkCreateShaderModule", context->CreateShaderModule(context->device, &info, NULL, &module->shader))) { tv_module_destroy(module); return NULL; }
  return module;
}

int tv_module_kernel(tv_module *module, uint32_t index, const char *entry, const tv_binding *bindings,
                     uint32_t binding_count, int has_parameters, uint32_t parameter_binding, uint32_t parameter_size) {
  if (!module || index >= module->kernel_count || !entry || !*entry || !binding_count) return module ? set_error(module->context, "invalid physical kernel %u", index) : 0;
  tv_context *context = module->context;
  tv_kernel *kernel = &module->kernels[index];
  kernel->entry = copy_string(entry); kernel->binding_count = binding_count;
  kernel->bindings = (tv_binding *)calloc(binding_count, sizeof(*kernel->bindings));
  if (!kernel->entry || !kernel->bindings) return set_error(context, "out of memory while creating physical kernel %u", index);
  memcpy(kernel->bindings, bindings, binding_count * sizeof(*bindings));
  kernel->has_parameters = has_parameters; kernel->parameter_binding = parameter_binding; kernel->parameter_size = parameter_size;
  if (has_parameters && (!parameter_size || parameter_size > context->properties.limits.maxUniformBufferRange)) return set_error(context, "kernel %s has an invalid parameter block", entry);
  for (uint32_t binding = 0; binding < binding_count; ++binding)
    if (bindings[binding].minimum_size > context->properties.limits.maxStorageBufferRange) return set_error(context, "kernel %s storage binding exceeds the device limit", entry);
  return 1;
}

int tv_module_prepare(tv_module *module, uint32_t index) {
  if (!module || index >= module->kernel_count) return module ? set_error(module->context, "invalid physical kernel %u", index) : 0;
  tv_context *context = module->context;
  tv_kernel *kernel = &module->kernels[index];
  if (kernel->pipeline) return 1;
  uint32_t binding_count = kernel->binding_count;
  VkDescriptorSetLayoutBinding *layout_bindings = (VkDescriptorSetLayoutBinding *)calloc(binding_count + (kernel->has_parameters ? 1 : 0), sizeof(*layout_bindings));
  if (!layout_bindings) return set_error(context, "out of memory while creating descriptor layout");
  for (uint32_t binding = 0; binding < binding_count; ++binding) {
    layout_bindings[binding] = (VkDescriptorSetLayoutBinding){ .binding = kernel->bindings[binding].binding, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
  }
  if (kernel->has_parameters) layout_bindings[binding_count] = (VkDescriptorSetLayoutBinding){ .binding = kernel->parameter_binding, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
  VkDescriptorSetLayoutCreateInfo descriptor_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = binding_count + (kernel->has_parameters ? 1 : 0), .pBindings = layout_bindings };
  int ok = result(context, "vkCreateDescriptorSetLayout", context->CreateDescriptorSetLayout(context->device, &descriptor_info, NULL, &kernel->descriptors));
  free(layout_bindings);
  if (!ok) return 0;
  VkPipelineLayoutCreateInfo layout_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &kernel->descriptors };
  if (!result(context, "vkCreatePipelineLayout", context->CreatePipelineLayout(context->device, &layout_info, NULL, &kernel->layout))) {
    context->DestroyDescriptorSetLayout(context->device, kernel->descriptors, NULL); kernel->descriptors = VK_NULL_HANDLE; return 0;
  }
  VkPipelineShaderStageCreateInfo stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = module->shader, .pName = kernel->entry };
  VkComputePipelineCreateInfo pipeline = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = kernel->layout };
  if (result(context, "vkCreateComputePipelines", context->CreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &pipeline, NULL, &kernel->pipeline))) return 1;
  context->DestroyPipelineLayout(context->device, kernel->layout, NULL); kernel->layout = VK_NULL_HANDLE;
  context->DestroyDescriptorSetLayout(context->device, kernel->descriptors, NULL); kernel->descriptors = VK_NULL_HANDLE;
  return 0;
}

void tv_module_destroy(tv_module *module) {
  if (!module) return;
  tv_context *context = module->context;
  if (context->device) context->DeviceWaitIdle(context->device);
  for (uint32_t index = 0; index < module->kernel_count; ++index) {
    tv_kernel *kernel = &module->kernels[index];
    if (kernel->pipeline) context->DestroyPipeline(context->device, kernel->pipeline, NULL);
    if (kernel->layout) context->DestroyPipelineLayout(context->device, kernel->layout, NULL);
    if (kernel->descriptors) context->DestroyDescriptorSetLayout(context->device, kernel->descriptors, NULL);
    free(kernel->bindings); free(kernel->entry);
  }
  if (module->shader) context->DestroyShaderModule(context->device, module->shader, NULL);
  free(module->kernels); free(module);
}

void tv_submission_barrier(tv_submission *submission) {
  if (!submission) return;
  transfer_barrier(submission->context, submission->commands,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

tv_submission *tv_submission_begin(tv_context *context, tv_submission *submission, uint32_t sets, uint32_t storage, uint32_t uniforms, uint32_t parameter_bytes) {
  if (!context || !sets) { if (context) set_error(context, "a submission must contain a dispatch"); return NULL; }
  uint32_t alignment = (uint32_t)context->properties.limits.minUniformBufferOffsetAlignment;
  if (!alignment) alignment = 1;
  uint64_t parameter_capacity = (uint64_t)parameter_bytes + (uniforms ? (uint64_t)(uniforms - 1) * (alignment - 1) : 0);
  if (parameter_capacity > UINT32_MAX) { set_error(context, "submission parameter arena is too large"); return NULL; }
  if (submission && (submission->context != context || submission->set_capacity < sets || submission->storage_capacity < storage ||
      submission->uniform_capacity < uniforms || submission->parameter_capacity < parameter_capacity)) {
    tv_submission_destroy(submission); submission = NULL;
  }
  if (!submission) {
    submission = (tv_submission *)calloc(1, sizeof(*submission));
    if (!submission) { set_error(context, "out of memory while creating a submission"); return NULL; }
    submission->context = context; submission->set_capacity = sets; submission->storage_capacity = storage;
    submission->uniform_capacity = uniforms; submission->parameter_capacity = (uint32_t)parameter_capacity;
    VkDescriptorPoolSize sizes[2]; uint32_t size_count = 0;
    if (storage) sizes[size_count++] = (VkDescriptorPoolSize){ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = storage };
    if (uniforms) sizes[size_count++] = (VkDescriptorPoolSize){ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = uniforms };
    VkDescriptorPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = sets, .poolSizeCount = size_count, .pPoolSizes = sizes };
    VkFenceCreateInfo fence = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (!result(context, "vkCreateDescriptorPool", context->CreateDescriptorPool(context->device, &pool, NULL, &submission->descriptors)) ||
        !allocate_commands(context, &submission->commands) ||
        !result(context, "vkCreateFence", context->CreateFence(context->device, &fence, NULL, &submission->fence))) { tv_submission_destroy(submission); return NULL; }
    if (uniforms) {
      submission->parameters = make_buffer(context, submission->parameter_capacity, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (!submission->parameters || !result(context, "vkMapMemory", context->MapMemory(context->device, submission->parameters->memory, 0,
          submission->parameters->allocation_size, 0, &submission->parameters->mapped))) { tv_submission_destroy(submission); return NULL; }
    }
  } else {
    if (!result(context, "vkResetDescriptorPool", context->ResetDescriptorPool(context->device, submission->descriptors, 0)) ||
        !result(context, "vkResetCommandBuffer", context->ResetCommandBuffer(submission->commands, 0)) ||
        !result(context, "vkResetFences", context->ResetFences(context->device, 1, &submission->fence))) { tv_submission_destroy(submission); return NULL; }
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (!result(context, "vkBeginCommandBuffer", context->BeginCommandBuffer(submission->commands, &begin))) { tv_submission_destroy(submission); return NULL; }
  }
  submission->parameter_offset = 0;
  tv_submission_barrier(submission);
  return submission;
}

int tv_submission_dispatch(tv_submission *submission, tv_module *module, uint32_t kernel_index,
                           const tv_resource *resources, uint32_t resource_count,
                           const uint8_t *parameters, uint32_t parameter_size,
                           uint32_t x, uint32_t y, uint32_t z) {
  if (!submission || !module || module->context != submission->context || kernel_index >= module->kernel_count || !x || !y || !z)
    return submission ? set_error(submission->context, "invalid dispatch") : 0;
  tv_context *context = submission->context; tv_kernel *kernel = &module->kernels[kernel_index];
  if (!tv_module_prepare(module, kernel_index) || resource_count != kernel->binding_count || parameter_size != (kernel->has_parameters ? kernel->parameter_size : 0))
    return set_error(context, "dispatch does not match physical kernel %s", kernel->entry ? kernel->entry : "<uninitialized>");
  if (x > context->properties.limits.maxComputeWorkGroupCount[0] || y > context->properties.limits.maxComputeWorkGroupCount[1] || z > context->properties.limits.maxComputeWorkGroupCount[2])
    return set_error(context, "dispatch exceeds Vulkan workgroup-count limits");
  VkDescriptorSetAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = submission->descriptors,
    .descriptorSetCount = 1, .pSetLayouts = &kernel->descriptors };
  VkDescriptorSet descriptors;
  if (!result(context, "vkAllocateDescriptorSets", context->AllocateDescriptorSets(context->device, &allocation, &descriptors))) return 0;
  uint32_t count = resource_count + (kernel->has_parameters ? 1 : 0);
  VkDescriptorBufferInfo *buffer_info = (VkDescriptorBufferInfo *)calloc(count, sizeof(*buffer_info));
  VkWriteDescriptorSet *writes = (VkWriteDescriptorSet *)calloc(count, sizeof(*writes));
  if (!buffer_info || !writes) { free(buffer_info); free(writes); return set_error(context, "out of memory while recording descriptors"); }
  for (uint32_t index = 0; index < resource_count; ++index) {
    if (resources[index].binding != kernel->bindings[index].binding || !resources[index].buffer || resources[index].size < kernel->bindings[index].minimum_size || resources[index].size > resources[index].buffer->size) {
      free(buffer_info); free(writes); return set_error(context, "dispatch resource %u has an invalid byte range", index);
    }
    buffer_info[index] = (VkDescriptorBufferInfo){ .buffer = resources[index].buffer->buffer, .offset = 0, .range = resources[index].size };
    writes[index] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptors,
      .dstBinding = resources[index].binding, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_info[index] };
  }
  if (kernel->has_parameters) {
    uint32_t alignment = (uint32_t)context->properties.limits.minUniformBufferOffsetAlignment;
    if (!alignment) alignment = 1;
    uint64_t offset = ((uint64_t)submission->parameter_offset + alignment - 1) / alignment * alignment;
    if (!submission->parameters || offset + parameter_size > submission->parameter_capacity) {
      free(buffer_info); free(writes); return 0;
    }
    memcpy((uint8_t *)submission->parameters->mapped + offset, parameters, parameter_size);
    submission->parameter_offset = (uint32_t)(offset + parameter_size);
    buffer_info[resource_count] = (VkDescriptorBufferInfo){ .buffer = submission->parameters->buffer, .offset = offset, .range = parameter_size };
    writes[resource_count] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptors,
      .dstBinding = kernel->parameter_binding, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_info[resource_count] };
  }
  context->UpdateDescriptorSets(context->device, count, writes, 0, NULL);
  free(buffer_info); free(writes);
  context->CmdBindPipeline(submission->commands, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
  context->CmdBindDescriptorSets(submission->commands, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->layout, 0, 1, &descriptors, 0, NULL);
  context->CmdDispatch(submission->commands, x, y, z);
  return 1;
}

int tv_submission_finish(tv_submission *submission) {
  if (!submission || submission->submitted) return 0;
  tv_context *context = submission->context;
  if (submission->parameters && !(submission->parameters->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = submission->parameters->memory, .offset = 0, .size = VK_WHOLE_SIZE };
    if (!result(context, "vkFlushMappedMemoryRanges", context->FlushMappedMemoryRanges(context->device, 1, &range))) return 0;
  }
  if (!result(context, "vkEndCommandBuffer", context->EndCommandBuffer(submission->commands))) return 0;
  if (!queue_commands(context, submission->commands, submission->fence)) return 0;
  submission->submitted = 1;
  return 1;
}

int tv_submission_wait(tv_submission *submission) {
  if (!submission || !submission->submitted) return 0;
  int ok = result(submission->context, "vkWaitForFences", submission->context->WaitForFences(submission->context->device, 1, &submission->fence, VK_TRUE, UINT64_MAX));
  if (ok) submission->submitted = 0;
  return ok;
}

void tv_submission_destroy(tv_submission *submission) {
  if (!submission) return;
  tv_context *context = submission->context;
  if (submission->submitted) tv_submission_wait(submission);
  if (submission->fence) context->DestroyFence(context->device, submission->fence, NULL);
  if (submission->commands) context->FreeCommandBuffers(context->device, context->command_pool, 1, &submission->commands);
  if (submission->descriptors) context->DestroyDescriptorPool(context->device, submission->descriptors, NULL);
  free_buffer(submission->parameters);
  free(submission);
}

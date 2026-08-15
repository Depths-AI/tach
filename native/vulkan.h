#ifndef TACH_VULKAN_H
#define TACH_VULKAN_H

#include <stddef.h>
#include <stdint.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

typedef struct tv_context tv_context;
typedef struct tv_module tv_module;
typedef struct tv_buffer tv_buffer;
typedef struct tv_submission tv_submission;

typedef struct {
  uint32_t binding;
  uint32_t minimum_size;
} tv_binding;

typedef struct {
  uint32_t binding;
  tv_buffer *buffer;
  uint32_t size;
} tv_resource;

tv_context *tv_open(int high_performance);
const char *tv_global_error(void);
const char *tv_error(tv_context *context);
const char *tv_device_name(tv_context *context);
uint32_t tv_vendor_id(tv_context *context);
uint32_t tv_device_type(tv_context *context);
void tv_close(tv_context *context);

tv_module *tv_module_create(tv_context *context, const uint8_t *spirv, size_t length, uint32_t kernel_count);
int tv_module_kernel(tv_module *module, uint32_t index, const char *entry, const tv_binding *bindings,
                     uint32_t binding_count, int has_parameters, uint32_t parameter_binding, uint32_t parameter_size);
void tv_module_destroy(tv_module *module);

tv_buffer *tv_buffer_create(tv_context *context, uint32_t size, const uint8_t *initial);
int tv_buffer_write(tv_buffer *buffer, const uint8_t *bytes, uint32_t length);
int tv_buffer_read(tv_buffer *buffer, uint8_t *output, uint32_t length);
void tv_buffer_destroy(tv_buffer *buffer);

tv_submission *tv_submission_begin(tv_context *context, tv_submission *reuse, uint32_t sets, uint32_t storage, uint32_t uniforms, uint32_t parameter_bytes);
int tv_submission_dispatch(tv_submission *submission, tv_module *module, uint32_t kernel,
                           const tv_resource *resources, uint32_t resource_count,
                           const uint8_t *parameters, uint32_t parameter_size,
                           uint32_t x, uint32_t y, uint32_t z);
void tv_submission_barrier(tv_submission *submission);
int tv_submission_finish(tv_submission *submission);
int tv_submission_wait(tv_submission *submission);
void tv_submission_destroy(tv_submission *submission);

#endif

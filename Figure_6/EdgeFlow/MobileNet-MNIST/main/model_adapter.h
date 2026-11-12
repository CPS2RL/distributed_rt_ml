#ifndef MODEL_ADAPTER_H
#define MODEL_ADAPTER_H
#include <stddef.h>
#include <stdint.h>
#define MAX_TIMED_LAYERS 32
int selected_model_ready(void);
int selected_model_infer(const uint8_t *pixels, size_t bytes, int device_id,
                         uint32_t timings_us[MAX_TIMED_LAYERS]);
int selected_model_layers(void);
const char *selected_model_layer_name(int index);
#endif

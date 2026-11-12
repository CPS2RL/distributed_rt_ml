#include "model_adapter.h"
#include <string.h>

#if EXP_MODEL_ID == 1
#include "distributed_lenet.h"
#include "model_weights.h"
#elif EXP_MODEL_ID == 2
#include "distributed_resnet.h"
#include "model_weights.h"
#elif EXP_MODEL_ID == 3
#include "distributed_mobilenet.h"
#include "model_weights.h"
#else
#error Unsupported model
#endif

int selected_model_ready(void) { return model_weights_get() != NULL; }
int selected_model_layers(void)
{
#if EXP_MODEL_ID == 1
    return 7;
#elif EXP_MODEL_ID == 2
    return 30;
#else
    return 29;
#endif
}

const char *selected_model_layer_name(int i)
{
#if EXP_MODEL_ID == 1
    static const char *names[] = {"Conv1", "Pool1", "Conv2", "Pool2", "FC1", "FC2", "FC3"};
    return (i >= 0 && i < 7) ? names[i] : "?";
#elif EXP_MODEL_ID == 2
    static const char *names[] = {
        "Stem", "B1C1", "B1C2", "B1Add", "B2C1", "B2C2", "B2Add",
        "B3C1", "B3C2", "B3Add", "B4C1", "B4C2", "B4Add",
        "B5C1", "B5C2", "B5Add", "B6C1", "B6C2", "B6Add",
        "B7C1", "B7C2", "B7Add", "B8C1", "B8C2", "B8Add",
        "B9C1", "B9C2", "B9Add", "GAP", "FC"
    };
    return (i >= 0 && i < 30) ? names[i] : "?";
#else
    static const char *names[] = {
        "Stem", "D1", "P1", "D2", "P2", "D3", "P3",
        "D4", "P4", "D5", "P5", "D6", "P6", "D7", "P7",
        "D8", "P8", "D9", "P9", "D10", "P10", "D11", "P11",
        "D12", "P12", "D13", "P13", "GAP", "FC"
    };
    return (i >= 0 && i < 29) ? names[i] : "?";
#endif
}

int selected_model_infer(const uint8_t *pixels, size_t bytes, int device_id,
                         uint32_t timings_us[MAX_TIMED_LAYERS])
{
    if (!selected_model_ready() || !timings_us) return -1;
    memset(timings_us, 0, sizeof(uint32_t) * MAX_TIMED_LAYERS);
    float logits[10];
#if EXP_MODEL_ID == 1
    return lenet_infer_local_timed(pixels, bytes, device_id,
                                  model_weights_get(), logits, timings_us);
#elif EXP_MODEL_ID == 2
    return resnet_infer_local_timed(pixels, bytes, device_id,
                                   model_weights_get(), logits, timings_us);
#else
    return mobilenet_infer_local_timed(pixels, bytes, device_id,
                                      model_weights_get(), logits, timings_us);
#endif
}

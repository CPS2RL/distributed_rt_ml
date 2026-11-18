#include "model_weights.h"

#if __has_include("model_weights_generated.h")
#include "model_weights_generated.h"
const LenetWeights *model_weights_get(void) { return &g_lenet_weights; }
const char *model_weights_description(void) { return MODEL_SOURCE_DESC; }
#else
const LenetWeights *model_weights_get(void) { return NULL; }
const char *model_weights_description(void) { return "MISSING: no trained weights embedded"; }
#endif

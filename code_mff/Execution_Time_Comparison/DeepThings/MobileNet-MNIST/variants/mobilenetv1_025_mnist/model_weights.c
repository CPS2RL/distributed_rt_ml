#include "model_weights.h"
#if __has_include("mobilenet_weights_generated.h")
#include "mobilenet_weights_generated.h"
const MobileWeights *model_weights_get(void) { return &g_mobilenet_weights; }
const char *model_weights_description(void) { return "TRAINED MobileNetV1-0.25/MNIST (BN fused)"; }
#else
const MobileWeights *model_weights_get(void) { return 0; }
const char *model_weights_description(void) { return "NO TRAINED WEIGHTS: export .pth first"; }
#endif

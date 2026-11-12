#include "model_weights.h"
#include "mobilenet_weights_generated.h"
const MobileWeights *model_weights_get(void) { return &g_mobilenet_weights; }
const char *model_weights_description(void) { return "TRAINED MobileNetV1-0.25/MNIST (BN fused)"; }

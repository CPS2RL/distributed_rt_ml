#include "model_weights.h"
#if __has_include("resnet_weights_generated.h")
#include "resnet_weights_generated.h"
const ResnetWeights *model_weights_get(void) { return &g_resnet_weights; }
const char *model_weights_description(void) { return "TRAINED ResNet20/CIFAR-10 (BN fused)"; }
#else
const ResnetWeights *model_weights_get(void) { return 0; }
const char *model_weights_description(void) { return "NO TRAINED WEIGHTS: export .pth first"; }
#endif

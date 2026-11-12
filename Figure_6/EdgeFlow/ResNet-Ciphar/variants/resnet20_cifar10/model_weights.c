#include "model_weights.h"
#include "resnet_weights_generated.h"
const ResnetWeights *model_weights_get(void) { return &g_resnet_weights; }
const char *model_weights_description(void) { return "TRAINED ResNet20/CIFAR-10 (BN fused)"; }

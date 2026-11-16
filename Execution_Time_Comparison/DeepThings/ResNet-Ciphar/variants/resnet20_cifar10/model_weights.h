#ifndef RESNET_MODEL_WEIGHTS_H
#define RESNET_MODEL_WEIGHTS_H
#include "distributed_resnet.h"
const ResnetWeights *model_weights_get(void);
const char *model_weights_description(void);
#endif

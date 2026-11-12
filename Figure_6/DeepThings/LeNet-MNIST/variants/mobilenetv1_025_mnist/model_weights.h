#ifndef MOBILENET_MODEL_WEIGHTS_H
#define MOBILENET_MODEL_WEIGHTS_H
#include "distributed_mobilenet.h"
const MobileWeights *model_weights_get(void);
const char *model_weights_description(void);
#endif

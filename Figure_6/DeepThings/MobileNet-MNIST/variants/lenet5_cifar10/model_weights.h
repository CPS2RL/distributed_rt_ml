#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H
#include "distributed_lenet.h"

/* Returns NULL until main/model_weights_generated.h is created.
 * This keeps the working ESP-NOW transfer buildable without model weights.
 */
const LenetWeights *model_weights_get(void);
const char *model_weights_description(void);

#endif

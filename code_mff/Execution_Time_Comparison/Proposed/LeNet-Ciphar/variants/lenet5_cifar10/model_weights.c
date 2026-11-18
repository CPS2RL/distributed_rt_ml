
#include "model_weights.h"
#include "model_weights_generated.h"

const LenetWeights *model_weights_get(void)
{
    return &g_lenet_weights;
}

const char *model_weights_description(void)
{
    return "TRAINED LeNet-5 MNIST";
}

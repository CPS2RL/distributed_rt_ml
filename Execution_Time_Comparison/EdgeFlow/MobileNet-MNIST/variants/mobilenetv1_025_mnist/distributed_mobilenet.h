#ifndef DISTRIBUTED_MOBILENET_H
#define DISTRIBUTED_MOBILENET_H
#include <stddef.h>
#include <stdint.h>

#define MOBILE_BLOCKS 13
/* PyTorch weights after BatchNorm fusion: standard [out,in,ky,kx],
 * depthwise [in,1,ky,kx], pointwise [out,in,1,1]. */
typedef struct {
    const float *weights;
    const float *bias;
    uint16_t in_channels;
    uint16_t out_channels;
    uint8_t kernel;
    uint8_t stride;
    uint8_t depthwise;
} MobileConv;

typedef struct {
    MobileConv depthwise;
    MobileConv pointwise;
} MobileBlock;

typedef struct {
    MobileConv stem;
    MobileBlock blocks[MOBILE_BLOCKS];
    const float *fc_weights; /* output-major [10, 256] */
    const float *fc_bias;    /* [10] */
} MobileWeights;

/* Input is *only* the assigned horizontal crop, row-major HWC bytes.
 * The crop runs as an independent image through all original MobileNetV1
 * layers, with local same-padding; there is no intermediate halo exchange.
 * Returns 0 on success, otherwise -1. 
 */
int mobilenet_infer_local(const uint8_t *pixels, size_t bytes,
                          int device_id, const MobileWeights *model,
                          float logits[10]);
int mobilenet_infer_local_timed(const uint8_t *pixels, size_t bytes,
                          int device_id, const MobileWeights *model,
                          float logits[10], uint32_t *us);
#endif

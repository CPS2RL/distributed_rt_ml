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

/* Input is the full image replicated to each ESP32. There is no
 * intermediate communication; each board independently computes the
 * full network with its trained weights. */
int mobilenet_infer_local(const uint8_t *pixels, size_t bytes,
                          int device_id, const MobileWeights *model,
                          float logits[10]);
int mobilenet_infer_local_timed(const uint8_t *pixels, size_t bytes,
                          int device_id, const MobileWeights *model,
                          float logits[10], uint32_t *us);
#endif

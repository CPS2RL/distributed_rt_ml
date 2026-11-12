#ifndef DISTRIBUTED_RESNET_H
#define DISTRIBUTED_RESNET_H
#include <stddef.h>
#include <stdint.h>

/* All convolution weights: PyTorch OIHW; all feature buffers: row-major HWC. */
typedef struct {
    const float *weights;   /* output, input, k, k */
    const float *bias;      /* BatchNorm folded, length output_channels */
    int input_channels;
    int output_channels;
    int kernel;
    int stride;
} ResnetConv;

typedef struct {
    ResnetConv conv1;
    ResnetConv conv2;
    ResnetConv projection; /* weights=NULL for identity shortcuts */
} ResnetBlock;

typedef struct {
    ResnetConv stem;
    ResnetBlock blocks[9]; /* 3 blocks per stage: 16, 32, 64 channels */
    const float *fc_weights; /* [10][64] output-major */
    const float *fc_bias;    /* [10] */
} ResnetWeights;

/* device_id: 0=rows 0..8, 1=9..17, 2=18..27. */
int resnet_input_rows(int device_id, int *start, int *end);
int resnet_infer_local(const uint8_t *stripe, size_t bytes, int device_id,
                       const ResnetWeights *model, float logits[10]);

int resnet_infer_local_timed(const uint8_t *pixels, size_t bytes, int device_id,
                       const ResnetWeights *model, float logits[10], uint32_t *us);
#endif

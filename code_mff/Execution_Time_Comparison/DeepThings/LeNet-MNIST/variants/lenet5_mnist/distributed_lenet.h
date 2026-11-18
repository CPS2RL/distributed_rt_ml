#ifndef DISTRIBUTED_LENET_H
#define DISTRIBUTED_LENET_H
#include <stddef.h>
#include <stdint.h>

/* Modern LeNet-5 variant: Conv5(1->6), MaxPool2, Conv5(6->16),
 * MaxPool2, Flatten(256), FC120, FC84, FC10. ReLU after conv/FC1/FC2.
 * Activations are HWC; convolution weights are [out][in][ky][kx];
 * FC weights are [out][in]. Pixel normalization is x/255.
 */
typedef struct {
    const float *c1_w, *c1_b;      /* 6*1*5*5, 6 */
    const float *c2_w, *c2_b;      /* 16*6*5*5, 16 */
    const float *f1_w, *f1_b;      /* 120*256, 120 */
    const float *f2_w, *f2_b;      /* 84*120, 84 */
    const float *f3_w, *f3_b;      /* 10*84, 10 */
} LenetWeights;

/* device_id=0,1,2. Returns input row range [start, end). */
int lenet_input_rows(int device_id, int *start, int *end);

/* Each ESP provides only its own (end-start)*28 grayscale uint8 pixels.
 * Computes no neighboring device rows at ANY stage; missing rows are zero.
 * Returns 0 on success and -1 on invalid arguments.
 * Not thread safe: uses fixed scratch buffers to avoid MCU stack overflow.
 */
int lenet_infer_local(const uint8_t *stripe, size_t bytes, int device_id,
                      const LenetWeights *weights, float logits[10]);
int lenet_argmax10(const float logits[10]);

int lenet_infer_local_timed(const uint8_t *stripe, size_t bytes, int device_id,
                      const LenetWeights *w, float logits[10], uint32_t *us);
#endif

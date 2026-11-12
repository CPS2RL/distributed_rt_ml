#ifndef DISTRIBUTED_LENET_H
#define DISTRIBUTED_LENET_H
#include <stddef.h>
#include <stdint.h>

/* LeNet-5 CIFAR-10 as defined by train_six_cnns.py:
 * 32x32x3 RGB -> Conv5 3->6 (28x28) -> MaxPool2 (14x14)
 * -> Conv5 6->16 (10x10) -> MaxPool2 (5x5)
 * -> HWC-flatten(400) -> FC120 -> FC84 -> FC10.
 * Convolution and FC use ReLU except final FC10.
 * All activations HWC; Conv weights OIHW; FC weights [out][in].
 */
typedef struct {
    const float *c1_w, *c1_b;   /* 6*3*5*5, 6 */
    const float *c2_w, *c2_b;   /* 16*6*5*5, 16 */
    const float *f1_w, *f1_b;   /* 120*400, 120 */
    const float *f2_w, *f2_b;   /* 84*120, 84 */
    const float *f3_w, *f3_b;   /* 10*84, 10 */
} LenetWeights;

/* Returns local input [first,end) row range, with exclusive end. */
int lenet_input_rows(int device_id, int *first, int *end);

/* local_rgb is RGB-interleaved row-major (end-first)*32*3 uint8 bytes.
 * Missing input/intermediate spatial rows are represented by zeros.
 * Each device computes only its assigned output rows at every layer.
 * No communication occurs between convolutional layers.
 * Uses static scratch buffers; NOT thread safe.
 * Returns 0 on success.
 */
int lenet_infer_local(const uint8_t *local_rgb, size_t bytes, int device_id,
                      const LenetWeights *weights, float logits[10]);
int lenet_argmax10(const float logits[10]);

int lenet_infer_local_timed(const uint8_t *stripe, size_t bytes, int device_id,
                      const LenetWeights *w, float logits[10], uint32_t *us);
#endif

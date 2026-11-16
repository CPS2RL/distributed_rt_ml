/* MobileNetV1-0.25: fully replicated global inference on each ESP32.
 * On 28/32-pixel inputs the final spatial height is one; disjoint
 * three-way final stripes are impossible. Here each ESP receives the full
 * input ONCE and redundantly recomputes all intermediates without radio.
 * BN is fused into the weights by tools/export_weights.py.
 * HWC activations. Weights remain in flash; scratch RAM ~16 KiB.
 */
#include "distributed_mobilenet.h"
#include <string.h>
#include "esp_timer.h"

#ifndef MOBILE_IMG_W
#error "Define MOBILE_IMG_W (28 or 32) in main/CMakeLists.txt"
#endif
#ifndef MOBILE_IMG_C
#error "Define MOBILE_IMG_C (1 or 3) in main/CMakeLists.txt"
#endif

#define SCRATCH_FLOATS 6144
static float s_work[2][SCRATCH_FLOATS];

static const int LOCAL_ROWS[3] = {
    MOBILE_IMG_W, MOBILE_IMG_W, MOBILE_IMG_W
};

static int valid(const MobileConv *c) {
    if (!c || !c->weights || !c->bias || !c->in_channels || !c->out_channels) return 0;
    if (c->kernel != 1 && c->kernel != 3) return 0;
    if (c->stride != 1 && c->stride != 2) return 0;
    if (c->depthwise && (c->kernel != 3 || c->out_channels != c->in_channels)) return 0;
    return 1;
}

static int layer(const float *src, float *dst, int h, int w,
                 const MobileConv *c, int *out_h, int *out_w) {
    if (!valid(c) || !src || !dst) return -1;
    const int oh = (h + c->stride - 1) / c->stride;
    const int ow = (w + c->stride - 1) / c->stride;
    const int ci = c->in_channels, co = c->out_channels;
    if ((size_t)oh * ow * co > SCRATCH_FLOATS) return -1;
    const int pad = c->kernel == 3 ? 1 : 0;
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            for (int oc = 0; oc < co; ++oc) {
                float sum = c->bias[oc];
                for (int ky = 0; ky < c->kernel; ++ky) {
                    int iy = y * c->stride + ky - pad;
                    if (iy < 0 || iy >= h) continue;
                    for (int kx = 0; kx < c->kernel; ++kx) {
                        int ix = x * c->stride + kx - pad;
                        if (ix < 0 || ix >= w) continue;
                        const float *activation = src + ((size_t)iy * w + ix) * ci;
                        if (c->depthwise) {
                            size_t wi = ((size_t)oc * 3 + ky) * 3 + kx;
                            sum += activation[oc] * c->weights[wi];
                        } else {
                            const float *kernel = c->weights +
                                 ((size_t)oc * ci * c->kernel * c->kernel) +
                                 ((size_t)ky * c->kernel + kx);
                            for (int ic = 0; ic < ci; ++ic) {
                                sum += activation[ic] *
                                    kernel[(size_t)ic * c->kernel * c->kernel];
                            }
                        }
                    }
                }
                if (sum < 0.0f) sum = 0.0f;  /* ReLU after BN */
                dst[((size_t)y * ow + x) * co + oc] = sum;
            }
        }
    }
    *out_h = oh;
    *out_w = ow;
    return 0;
}

int mobilenet_infer_local_timed(const uint8_t *pixels, size_t bytes,
                          int device_id, const MobileWeights *model,
                          float logits[10], uint32_t *us) {
    if (!pixels || !model || !logits || device_id < 0 || device_id > 2 ||
        !model->fc_weights || !model->fc_bias || !valid(&model->stem)) return -1;
    const int rows = LOCAL_ROWS[device_id];
    const int width = MOBILE_IMG_W;
    const int channels = MOBILE_IMG_C;
    const size_t required = (size_t)rows * width * channels;
    if (bytes != required || required > SCRATCH_FLOATS ||
        model->stem.in_channels != channels || model->stem.out_channels != 8 ||
        model->stem.kernel != 3 || model->stem.stride != 2 ||
        model->stem.depthwise) return -1;
    for (size_t i = 0; i < required; ++i) s_work[0][i] = pixels[i] / 255.0f;
    int active = 0, h = rows, w = width, c = channels;
    const MobileConv *conv = &model->stem;
    int next_h, next_w;
    int64_t tick = esp_timer_get_time();
    if (layer(s_work[active], s_work[1-active], h, w,
              conv, &next_h, &next_w) != 0) return -1;
    if (us) us[0] = (uint32_t)(esp_timer_get_time() - tick);
    active = 1-active; h = next_h; w = next_w; c = conv->out_channels;
    for (int i = 0; i < MOBILE_BLOCKS; ++i) {
        const MobileBlock *b = &model->blocks[i];
        if (b->depthwise.in_channels != c || b->depthwise.out_channels != c ||
            !b->depthwise.depthwise || b->depthwise.kernel != 3 ||
            b->pointwise.in_channels != c || b->pointwise.kernel != 1 ||
            b->pointwise.depthwise || b->pointwise.stride != 1) return -1;
        tick = esp_timer_get_time();
        if (layer(s_work[active], s_work[1-active], h, w,
                  &b->depthwise, &next_h, &next_w) != 0) return -1;
        if (us) us[1 + 2 * i] = (uint32_t)(esp_timer_get_time() - tick);
        active = 1-active; h = next_h; w = next_w;
        tick = esp_timer_get_time();
        if (layer(s_work[active], s_work[1-active], h, w,
                  &b->pointwise, &next_h, &next_w) != 0) return -1;
        if (us) us[2 + 2 * i] = (uint32_t)(esp_timer_get_time() - tick);
        active = 1-active; h = next_h; w = next_w;
        c = b->pointwise.out_channels;
    }
    if (c != 256 || h < 1 || w < 1) return -1;
    tick = esp_timer_get_time();
    float mean[256] = {0};
    const int n = h * w;
    for (int pos = 0; pos < n; ++pos)
        for (int k = 0; k < 256; ++k)
            mean[k] += s_work[active][(size_t)pos * 256 + k];
    for (int k = 0; k < 256; ++k) mean[k] /= (float)n;
    if (us) us[27] = (uint32_t)(esp_timer_get_time() - tick);
    tick = esp_timer_get_time();
    for (int cls = 0; cls < 10; ++cls) {
        float sum = model->fc_bias[cls];
        for (int k = 0; k < 256; ++k) {
            sum += mean[k] * model->fc_weights[(size_t)cls * 256 + k];
        }
        logits[cls] = sum;
    }
    if (us) us[28] = (uint32_t)(esp_timer_get_time() - tick);
    return 0;
}

int mobilenet_infer_local(const uint8_t *pixels, size_t bytes,
                          int device_id, const MobileWeights *model,
                          float logits[10])
{
    return mobilenet_infer_local_timed(pixels, bytes, device_id, model, logits, NULL);
}

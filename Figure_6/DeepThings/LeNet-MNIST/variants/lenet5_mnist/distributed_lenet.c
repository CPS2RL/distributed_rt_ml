#include "distributed_lenet.h"
#include <string.h>
#include "esp_timer.h"
#include "partition_config.h"

/* For each stage: boundaries for device 0 / device 1 / device 2.
 * Output stripes at Conv2 are deliberately aligned with MaxPool2.
 *            Input Conv1 Pool1 Conv2 Pool2
 * ESP32 #1:   0:9   0:8  0:4   0:2   0:1
 * ESP32 #2:  9:18  8:16  4:8   2:4   1:2
 * ESP32 #3: 18:28 16:24 8:12   4:8   2:4
 */
/* Per-layer boundaries are defined only in partition_config.h. */
/* Ranges derive from PART_LENET final-layer ownership. */

/* Static storage: ~26 KB scratch; single inference task per board. */
static float input[28 * 28];
static float c1[24 * 24 * 6];
static float p1[12 * 12 * 6];
static float c2[8 * 8 * 16];
static float p2[4 * 4 * 16];
static float fc1[120], fc2[84];

int lenet_input_rows(int device_id, int *start, int *end) {
    if (device_id < 0 || device_id > 2 || !start || !end) return -1;
    PartRange plan[5]; redundant_lenet_plan(device_id, plan);
    *start = plan[0].lo;
    *end = plan[0].hi;
    return 0;
}

/* Valid 5x5 convolution, output in expanded [first,last). HWC.
 * Backward expansion ensures all needed rows were computed locally. */
static void conv5_relu(const float *x, int hi, int wi, int ci,
                       const float *w, const float *b, int co,
                       int first, int last, float *y) {
    const int ho = hi - 4, wo = wi - 4;
    memset(y, 0, (size_t)ho * wo * co * sizeof(*y));
    for (int row = first; row < last; ++row) {
        for (int col = 0; col < wo; ++col) {
            for (int oc = 0; oc < co; ++oc) {
                float sum = b[oc];
                for (int ic = 0; ic < ci; ++ic) {
                    for (int kr = 0; kr < 5; ++kr) {
                        for (int kc = 0; kc < 5; ++kc) {
                            const size_t ix = ((size_t)(row + kr) * wi + (col + kc)) * ci + ic;
                            const size_t iw = (((size_t)oc * ci + ic) * 5 + kr) * 5 + kc;
                            sum += x[ix] * w[iw];
                        }
                    }
                }
                y[((size_t)row * wo + col) * co + oc] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
}

static void maxpool2(const float *x, int hi, int wi, int channels,
                     int first, int last, float *y) {
    const int ho = hi / 2, wo = wi / 2;
    memset(y, 0, (size_t)ho * wo * channels * sizeof(*y));
    for (int row = first; row < last; ++row) {
        for (int col = 0; col < wo; ++col) {
            for (int ch = 0; ch < channels; ++ch) {
                float maxv = x[((size_t)(2 * row) * wi + 2 * col) * channels + ch];
                for (int dr = 0; dr < 2; ++dr) {
                    for (int dc = 0; dc < 2; ++dc) {
                        float v = x[((size_t)(2 * row + dr) * wi + (2 * col + dc)) * channels + ch];
                        if (v > maxv) maxv = v;
                    }
                }
                y[((size_t)row * wo + col) * channels + ch] = maxv;
            }
        }
    }
}

static void dense(const float *x, int ni, const float *w,
                  const float *b, int no, int relu, float *y) {
    for (int o = 0; o < no; ++o) {
        float sum = b[o];
        for (int i = 0; i < ni; ++i) sum += w[(size_t)o * ni + i] * x[i];
        y[o] = (relu && sum < 0.0f) ? 0.0f : sum;
    }
}

int lenet_infer_local_timed(const uint8_t *stripe, size_t bytes, int device_id,
                      const LenetWeights *w, float logits[10], uint32_t *us) {
    if (!stripe || !w || !logits || device_id < 0 || device_id > 2 ||
        !w->c1_w || !w->c1_b || !w->c2_w || !w->c2_b ||
        !w->f1_w || !w->f1_b || !w->f2_w || !w->f2_b ||
        !w->f3_w || !w->f3_b) return -1;

    PartRange ranges[5]; redundant_lenet_plan(device_id, ranges);
    const int r0 = ranges[0].lo, r1 = ranges[0].hi;
    if (bytes != (size_t)(r1 - r0) * 28) return -1;

    memset(input, 0, sizeof(input));
    for (int r = r0; r < r1; ++r)
        for (int col = 0; col < 28; ++col)
            input[r * 28 + col] = stripe[(r - r0) * 28 + col] / 255.0f;


    int64_t tick_0 = esp_timer_get_time();
    conv5_relu(input, 28, 28, 1, w->c1_w, w->c1_b, 6,
               ranges[1].lo, ranges[1].hi, c1);
    if (us) us[0] = (uint32_t)(esp_timer_get_time() - tick_0);

    int64_t tick_1 = esp_timer_get_time();
    maxpool2(c1, 24, 24, 6,
             ranges[2].lo, ranges[2].hi, p1);
    if (us) us[1] = (uint32_t)(esp_timer_get_time() - tick_1);

    int64_t tick_2 = esp_timer_get_time();
    conv5_relu(p1, 12, 12, 6, w->c2_w, w->c2_b, 16,
               ranges[3].lo, ranges[3].hi, c2);
    if (us) us[2] = (uint32_t)(esp_timer_get_time() - tick_2);

    int64_t tick_3 = esp_timer_get_time();
    maxpool2(c2, 8, 8, 16,
             ranges[4].lo, ranges[4].hi, p2);

    if (us) us[3] = (uint32_t)(esp_timer_get_time() - tick_3);
    /* Full-shape 256-element FC input, but only this device's spatial
     * stripe has values. Other rows are zero (and never computed).
     */
    int64_t tick_4 = esp_timer_get_time();
    dense(p2, 256, w->f1_w, w->f1_b, 120, 1, fc1);
    if (us) us[4] = (uint32_t)(esp_timer_get_time() - tick_4);
    int64_t tick_5 = esp_timer_get_time();
    dense(fc1, 120, w->f2_w, w->f2_b, 84, 1, fc2);
    if (us) us[5] = (uint32_t)(esp_timer_get_time() - tick_5);
    int64_t tick_6 = esp_timer_get_time();
    dense(fc2, 84, w->f3_w, w->f3_b, 10, 0, logits);
    if (us) us[6] = (uint32_t)(esp_timer_get_time() - tick_6);
    return 0;
}

int lenet_infer_local(const uint8_t *stripe, size_t bytes, int device_id,
                      const LenetWeights *w, float logits[10])
{
    return lenet_infer_local_timed(stripe, bytes, device_id, w, logits, NULL);
}

int lenet_argmax10(const float logits[10]) {
    if (!logits) return -1;
    int best = 0;
    for (int i = 1; i < 10; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

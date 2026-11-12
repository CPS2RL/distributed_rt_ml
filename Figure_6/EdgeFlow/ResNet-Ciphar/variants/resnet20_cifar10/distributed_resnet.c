/*
 * ResNet-20 CIFAR-10, 3 horizontal partitions, no intermediate halo exchange.
 *
 * Stem: 3x3, 3->16, stride 1, pad 1, BN folded, ReLU
 * Stages: 3 residual blocks each, channels 16 / 32 / 64.
 * Stage 2 and 3 start with stride 2 and a fused 1x1 projection shortcut.
 * GAP over each device's owned rows of the last 8x8x64 map, then shared FC.
 *
 * All C activation tensors are HWC and keep only the assigned global rows.
 * Convolution reads zeros for rows outside the local stripe (zero-halo).
 * Three reused scratch buffers ~66KiB total. Weights remain in flash.
 * Non-reentrant: only one inference call per ESP32 at a time.
 */
#include "distributed_resnet.h"
#include <string.h>
#include "esp_timer.h"
#include "partition_config.h"
#include "halo_exchange.h"

#define MAX_ACT_FLOATS 6144

static float g_work[3][MAX_ACT_FLOATS];
static float g_proj[MAX_ACT_FLOATS];

/* Global row boundaries: 32x32 stem/stage1, 16x16 stage2, 8x8 stage3. */
#define ROW_BOUNDS PART_RESNET

typedef struct {
    const float *data;
    int first;     /* global first row */
    int last;      /* global exclusive end row */
    int width;
    int channels;
} Stripe;

int resnet_input_rows(int id, int *start, int *end)
{
    if (id < 0 || id >= 3 || !start || !end) return -1;
    *start = ROW_BOUNDS[0][id];
    *end = ROW_BOUNDS[0][id + 1];
    return 0;
}

static int is_valid_conv(const ResnetConv *c)
{
    return c && c->weights && c->bias &&
           c->input_channels > 0 && c->output_channels > 0 &&
           (c->kernel == 1 || c->kernel == 3) &&
           (c->stride == 1 || c->stride == 2);
}

/* Compute only rows first..last. All other global rows are implicitly zero.
 * Bias includes the folded BatchNorm offset. ReLU is optional.
 */
static void convolution(const Stripe *in, const ResnetConv *layer,
                        int output_width, int first, int last,
                        int relu, float *dest)
{
    const int icount = layer->input_channels;
    const int ocount = layer->output_channels;
    const int k = layer->kernel;
    const int stride = layer->stride;
    const int pad = k / 2;

    for (int oy = first; oy < last; ++oy) {
        for (int ox = 0; ox < output_width; ++ox) {
            for (int oc = 0; oc < ocount; ++oc) {
                float sum = layer->bias[oc];
                for (int ky = 0; ky < k; ++ky) {
                    const int iy = oy * stride + ky - pad;
                    const float *rptr = halo_row(in->data, iy,
                        in->first, in->last, in->width, in->channels);
                    if (!rptr) continue;
                    for (int kx = 0; kx < k; ++kx) {
                        const int ix = ox * stride + kx - pad;
                        if (ix < 0 || ix >= in->width) continue;
                        const float *x = rptr + (size_t)ix * icount;
                        for (int ic = 0; ic < icount; ++ic) {
                            const size_t wi =
                                (((size_t)oc * icount + ic) * k + ky) * k + kx;
                            sum += x[ic] * layer->weights[wi];
                        }
                    }
                }
                if (relu && sum < 0.0f) sum = 0.0f;
                dest[((size_t)(oy - first) * output_width + ox) * ocount + oc] = sum;
            }
        }
    }
}

/* 1x1 BN-fused projection: output sample (oy, ox), may read zero if a local
 * input row is unavailable. For this model projections have stride 2.
 */
static float projection_value(const Stripe *in, const ResnetConv *projection,
                              int oy, int ox, int oc)
{
    float sum = projection->bias[oc];
    const int iy = oy * projection->stride;
    const int ix = ox * projection->stride;
    if (ix >= in->width) return sum;
    const float *inputrow = halo_row(in->data, iy, in->first, in->last,
                                     in->width, in->channels);
    if (!inputrow) return sum;
    const float *x = inputrow + (size_t)ix * in->channels;
    const float *w = projection->weights + (size_t)oc * in->channels;
    for (int ic = 0; ic < in->channels; ++ic) sum += x[ic] * w[ic];
    return sum;
}

int resnet_infer_local_timed(const uint8_t *pixels, size_t bytes, int device_id,
                       const ResnetWeights *model, float logits[10], uint32_t *us)
{
    if (!pixels || !model || !logits || device_id < 0 || device_id > 2 ||
        !model->fc_weights || !model->fc_bias || !is_valid_conv(&model->stem))
        return -1;

    const int first = ROW_BOUNDS[0][device_id];
    const int last = ROW_BOUNDS[0][device_id + 1];
    if (bytes != (size_t)(last - first) * 32 * 3) return -1;

    /* Local input pixels are uint8, converted to float in [0,1]. */
    for (size_t i = 0; i < bytes; ++i) g_work[0][i] = (float)pixels[i] / 255.0f;

    Stripe current = {g_work[0], first, last, 32, 3};
    if (halo_prepare(&(HaloSpec){.step=0, .source_height=EXP_IMG_W,
          .width=EXP_IMG_W, .channels=EXP_IMG_C, .kernel=3,
          .stride=1, .pad=1, .source_bounds=ROW_BOUNDS[0],
          .destination_bounds=ROW_BOUNDS[0], .src=g_work[0],
          .compact=1}) != 0) return -1;
    int64_t tick = esp_timer_get_time();
    convolution(&current, &model->stem, 32, first, last, 1, g_work[1]);
    if (us) us[0] = (uint32_t)(esp_timer_get_time() - tick);
    int active = 1;
    current = (Stripe){g_work[active], first, last, 32, 16};

    for (int b = 0; b < 9; ++b) {
        const ResnetBlock *block = &model->blocks[b];
        if (!is_valid_conv(&block->conv1) || !is_valid_conv(&block->conv2))
            return -1;
        if (block->conv1.input_channels != current.channels) return -1;
        const int stage = b / 3;
        const int width = (stage == 0 ? 32 : stage == 1 ? 16 : 8);
        const int output_first = ROW_BOUNDS[stage][device_id];
        const int output_last = ROW_BOUNDS[stage][device_id + 1];
        const int channels = block->conv1.output_channels;
        const size_t output_count =
            (size_t)(output_last - output_first) * width * channels;
        if (output_count > MAX_ACT_FLOATS) return -1;

        const int tmp = (active + 1) % 3;
        const int tmp2 = (active + 2) % 3;
        const int input_h = current.width;
        const uint8_t *input_bounds = (b==3) ? ROW_BOUNDS[0] :
                                       (b==6) ? ROW_BOUNDS[1] : ROW_BOUNDS[stage];
        if (halo_prepare(&(HaloSpec){.step=1+b*3, .source_height=input_h,
              .width=current.width, .channels=current.channels, .kernel=3,
              .stride=block->conv1.stride, .pad=1,
              .source_bounds=input_bounds, .destination_bounds=ROW_BOUNDS[stage],
              .src=(float*)current.data, .compact=1}) != 0) return -1;
        /* Compute projection while the original skip input halo is valid. */
        uint32_t proj_us=0;
        if (block->projection.weights) {
            int64_t began_proj=esp_timer_get_time();
            for (int oy=output_first;oy<output_last;oy++)
                for (int ox=0;ox<width;ox++)
                    for (int oc=0;oc<channels;oc++) {
                        size_t index=((size_t)(oy-output_first)*width+ox)*channels+oc;
                        g_proj[index]=projection_value(&current,&block->projection,
                                                        oy,ox,oc);
                    }
            proj_us=(uint32_t)(esp_timer_get_time()-began_proj);
        }
        tick = esp_timer_get_time();
        convolution(&current, &block->conv1, width,
                    output_first, output_last, 1, g_work[tmp]);
        if (us) us[1 + b * 3] = (uint32_t)(esp_timer_get_time() - tick);
        Stripe midway = {g_work[tmp], output_first, output_last, width, channels};
        if (halo_prepare(&(HaloSpec){.step=2+b*3, .source_height=width,
              .width=width, .channels=channels, .kernel=3, .stride=1, .pad=1,
              .source_bounds=ROW_BOUNDS[stage],
              .destination_bounds=ROW_BOUNDS[stage],
              .src=g_work[tmp], .compact=1}) != 0) return -1;
        tick = esp_timer_get_time();
        convolution(&midway, &block->conv2, width,
                    output_first, output_last, 0, g_work[tmp2]);
        if (us) us[2 + b * 3] = (uint32_t)(esp_timer_get_time() - tick);

        /* Skip connection plus second convolution; final ReLU.
         * Conv1 buffer can be reused now that Conv2 has finished.
         */
        tick = esp_timer_get_time();
        for (int oy = output_first; oy < output_last; ++oy) {
            for (int ox = 0; ox < width; ++ox) {
                for (int oc = 0; oc < channels; ++oc) {
                    const size_t out_i =
                        ((size_t)(oy - output_first) * width + ox) * channels + oc;
                    float skip;
                    if (block->projection.weights) {
                        skip = g_proj[out_i];
                    } else {
                        /* Identity branch: same global rows and tensor dimensions. */
                        skip = current.data[out_i];
                    }
                    float result = g_work[tmp2][out_i] + skip;
                    g_work[tmp][out_i] = result > 0.0f ? result : 0.0f;
                }
            }
        }
        if (us) us[3 + b * 3] =
            (uint32_t)(esp_timer_get_time() - tick) + proj_us;
        active = tmp;
        current = (Stripe){g_work[active], output_first, output_last, width, channels};
    }

    if (current.width != 8 || current.channels != 64 ||
        current.last <= current.first) return -1;

    /* Device-local global average pool over OWNED final rows only. */
    tick = esp_timer_get_time();
    float pooled[64] = {0};
    const int count = (current.last - current.first) * current.width;
    for (int p = 0; p < count; ++p)
        for (int c = 0; c < 64; ++c)
            pooled[c] += current.data[(size_t)p * 64 + c];
    for (int c = 0; c < 64; ++c) pooled[c] /= (float)count;

    if (us) us[28] = (uint32_t)(esp_timer_get_time() - tick);

    /* Classifier of the full-image trained model, reused unchanged. */
    tick = esp_timer_get_time();
    for (int cls = 0; cls < 10; ++cls) {
        float sum = model->fc_bias[cls];
        for (int c = 0; c < 64; ++c)
            sum += pooled[c] * model->fc_weights[(size_t)cls * 64 + c];
        logits[cls] = sum;
    }
    if (us) us[29] = (uint32_t)(esp_timer_get_time() - tick);
    return 0;
}

int resnet_infer_local(const uint8_t *pixels, size_t bytes, int device_id,
                       const ResnetWeights *model, float logits[10])
{
    return resnet_infer_local_timed(pixels, bytes, device_id, model, logits, NULL);
}

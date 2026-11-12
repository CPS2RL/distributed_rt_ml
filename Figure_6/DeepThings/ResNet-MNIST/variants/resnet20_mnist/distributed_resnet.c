/* ResNet-20: communication-free redundant execution, SRAM-optimized.
 * Uses one feature map plus a 3-row ring for Conv1, and one Conv2 output row.
 * Conv2 results and shortcut are combined into the feature map in place.
 * No intermediate communication and no requirement for PSRAM.
 *
 * Stable timing: 30 entries [Stem, B1C1, B1C2, B1Add,..., GAP, FC].
 */
#include "distributed_resnet.h"
#include "partition_config.h"
#include <string.h>
#include <limits.h>
#include "esp_timer.h"

/* Each stage has H * 16 floats per feature row:
 * stage1 H x 16, stage2 (H/2) x 32, stage3 (H/4) x 64.
 */
#define MAX_ROW_FLOATS (EXP_IMG_W * 16)
#define MAX_ACT_FLOATS (EXP_IMG_W * MAX_ROW_FLOATS)
static float g_feature[MAX_ACT_FLOATS];
static float g_input[EXP_IMG_W * EXP_IMG_W * EXP_IMG_C];
static float g_middle_ring[3][MAX_ROW_FLOATS];
static float g_conv2_row[MAX_ROW_FLOATS];
static float g_shortcut_row[MAX_ROW_FLOATS];

typedef struct {
    const float *data;
    int lo, hi, w, channels;
} Stripe;

static int valid_conv(const ResnetConv *c) {
    return c && c->weights && c->bias && c->input_channels > 0 &&
           c->output_channels > 0 && (c->kernel == 1 || c->kernel == 3) &&
           (c->stride == 1 || c->stride == 2);
}

static const float *local_row(const Stripe *in, int y) {
    if (y < in->lo || y >= in->hi) return NULL;
    return in->data + (size_t)(y - in->lo) * in->w * in->channels;
}

/* Pre-normalized FP32 input: preprocessing stays outside layer timing. */
static int stem_from_float(const float *pixels, PartRange input,
                            PartRange out, int h, int channels,
                            const ResnetConv *c) {
    if (c->kernel != 3 || c->stride != 1 || c->input_channels != channels ||
        c->output_channels != 16) return -1;
    if ((size_t)(out.hi - out.lo) * h * 16 > MAX_ACT_FLOATS) return -1;
    for (int oy = out.lo; oy < out.hi; ++oy)
        for (int ox = 0; ox < h; ++ox)
            for (int oc = 0; oc < 16; ++oc) {
                float sum = c->bias[oc];
                for (int ky = 0; ky < 3; ++ky) {
                    int iy = oy + ky - 1;
                    if (iy < 0 || iy >= h) continue;
                    if (iy < input.lo || iy >= input.hi) return -1;
                    const float *row = pixels +
                          (size_t)(iy - input.lo) * h * channels;
                    for (int kx = 0; kx < 3; ++kx) {
                        int ix = ox + kx - 1;
                        if (ix < 0 || ix >= h) continue;
                        const float *x = row + (size_t)ix * channels;
                        for (int ic = 0; ic < channels; ++ic) {
                            size_t wi = (((size_t)oc * channels + ic) * 3 + ky) * 3 + kx;
                            sum += x[ic] * c->weights[wi];
                        }
                    }
                }
                if (sum < 0.0f) sum = 0.0f;
                g_feature[((size_t)(oy - out.lo) * h + ox) * 16 + oc] = sum;
            }
    return 0;
}

/* One output row of Conv1, into the rolling ring. */
static int compute_conv1_row(const Stripe *in, const ResnetConv *c,
                             int out_w, int oy, float *dst) {
    const int ci = c->input_channels, co = c->output_channels;
    if (out_w * co > MAX_ROW_FLOATS || c->kernel != 3) return -1;
    for (int ox = 0; ox < out_w; ++ox)
        for (int oc = 0; oc < co; ++oc) {
            float sum = c->bias[oc];
            for (int ky = 0; ky < 3; ++ky) {
                int iy = oy * c->stride + ky - 1;
                if (iy < 0 || iy >= in->w) continue;
                const float *row = local_row(in, iy);
                if (!row) return -1;
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = ox * c->stride + kx - 1;
                    if (ix < 0 || ix >= in->w) continue;
                    const float *x = row + (size_t)ix * ci;
                    for (int ic = 0; ic < ci; ++ic) {
                        size_t wi = (((size_t)oc * ci + ic) * 3 + ky) * 3 + kx;
                        sum += x[ic] * c->weights[wi];
                    }
                }
            }
            dst[(size_t)ox * co + oc] = sum > 0.0f ? sum : 0.0f;
        }
    return 0;
}

/* One Conv2 output row; Conv1 ring must contain needed middle rows. */
static int compute_conv2_row(const ResnetConv *c, int w, int oy,
                             const int ring_tag[3], float *dst) {
    const int ci = c->input_channels, co = c->output_channels;
    if (c->kernel != 3 || c->stride != 1 || w * co > MAX_ROW_FLOATS) return -1;
    for (int ox = 0; ox < w; ++ox)
        for (int oc = 0; oc < co; ++oc) {
            float sum = c->bias[oc];
            for (int ky = 0; ky < 3; ++ky) {
                int iy = oy + ky - 1;
                if (iy < 0 || iy >= w) continue;
                int slot = iy % 3;
                if (ring_tag[slot] != iy) return -1;
                const float *row = g_middle_ring[slot];
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = ox + kx - 1;
                    if (ix < 0 || ix >= w) continue;
                    const float *x = row + (size_t)ix * ci;
                    for (int ic = 0; ic < ci; ++ic) {
                        size_t wi = (((size_t)oc * ci + ic) * 3 + ky) * 3 + kx;
                        sum += x[ic] * c->weights[wi];
                    }
                }
            }
            dst[(size_t)ox * co + oc] = sum;
        }
    return 0;
}

/* Produce one block, in place. The old input must survive long enough to
 * compute the next Conv1 halo rows and this output row's shortcut. */
static int process_block(Stripe *in, const ResnetBlock *b,
                         PartRange mid, PartRange out, int out_w,
                         uint32_t *conv1_us, uint32_t *conv2_us,
                         uint32_t *add_us) {
    if (!valid_conv(&b->conv1) || !valid_conv(&b->conv2) ||
        b->conv1.kernel != 3 || b->conv2.kernel != 3 ||
        b->conv1.input_channels != in->channels ||
        b->conv2.input_channels != b->conv1.output_channels ||
        b->conv2.output_channels != b->conv1.output_channels ||
        b->conv2.stride != 1) return -1;
    const int out_ch = b->conv2.output_channels;
    const int out_row_n = out_w * out_ch;
    if (out_row_n > MAX_ROW_FLOATS ||
        (size_t)(out.hi - out.lo) * out_row_n > MAX_ACT_FLOATS) return -1;
    if (b->projection.weights) {
        if (!valid_conv(&b->projection) || b->projection.kernel != 1 ||
            b->projection.input_channels != in->channels ||
            b->projection.output_channels != out_ch ||
            b->projection.stride != b->conv1.stride) return -1;
    } else if (in->w != out_w || in->channels != out_ch) {
        return -1;
    }
    int ring_tag[3] = {-1, -1, -1};
    uint64_t a_us = 0, b_us = 0, c_us = 0;

    for (int oy = out.lo; oy < out.hi; ++oy) {
        /* Compute all needed middle rows BEFORE overwriting input features. */
        int64_t tick = esp_timer_get_time();
        for (int my = oy - 1; my <= oy + 1; ++my) {
            if (my < 0 || my >= out_w) continue;
            if (my < mid.lo || my >= mid.hi) return -1;
            int slot = my % 3;
            if (ring_tag[slot] != my) {
                if (compute_conv1_row(in, &b->conv1, out_w, my,
                                      g_middle_ring[slot]) != 0) return -1;
                ring_tag[slot] = my;
            }
        }
        a_us += (uint64_t)(esp_timer_get_time() - tick);
        tick = esp_timer_get_time();
        if (compute_conv2_row(&b->conv2, out_w, oy,
                              ring_tag, g_conv2_row) != 0) return -1;
        b_us += (uint64_t)(esp_timer_get_time() - tick);

        tick = esp_timer_get_time();
        if (b->projection.weights) {
            const ResnetConv *p = &b->projection;
            const float *source = local_row(in, oy * p->stride);
            if (!source) return -1;
            for (int ox = 0; ox < out_w; ++ox) {
                const float *x = source + (size_t)(ox * p->stride) * p->input_channels;
                for (int oc = 0; oc < out_ch; ++oc) {
                    float v = p->bias[oc];
                    for (int ic = 0; ic < p->input_channels; ++ic)
                        v += x[ic] * p->weights[(size_t)oc * p->input_channels + ic];
                    g_shortcut_row[(size_t)ox * out_ch + oc] = v;
                }
            }
        } else {
            const float *source = local_row(in, oy);
            if (!source) return -1;
            memcpy(g_shortcut_row, source, (size_t)out_row_n * sizeof(float));
        }
        /* g_feature will be repacked to the new output stripe's start. */
        float *dest = g_feature + (size_t)(oy - out.lo) * out_row_n;
        for (int j = 0; j < out_row_n; ++j) {
            float v = g_conv2_row[j] + g_shortcut_row[j];
            dest[j] = v > 0.0f ? v : 0.0f;
        }
        c_us += (uint64_t)(esp_timer_get_time() - tick);
    }
    *in = (Stripe){g_feature, out.lo, out.hi, out_w, out_ch};
    if (conv1_us) *conv1_us = (a_us > UINT32_MAX) ? UINT32_MAX : (uint32_t)a_us;
    if (conv2_us) *conv2_us = (b_us > UINT32_MAX) ? UINT32_MAX : (uint32_t)b_us;
    if (add_us) *add_us = (c_us > UINT32_MAX) ? UINT32_MAX : (uint32_t)c_us;
    return 0;
}

int resnet_input_rows(int id,int *lo,int *hi) {
    if(id<0 || id>=3 || !lo || !hi) return -1;
    RedundantResnetPlan p=redundant_resnet_plan(id);
    *lo=p.input.lo; *hi=p.input.hi; return 0;
}

int resnet_infer_local_timed(const uint8_t *pixels, size_t bytes, int id,
                            const ResnetWeights *m, float logits[10],
                            uint32_t *us) {
    if (!pixels || !m || !logits || id < 0 || id > 2 ||
        !valid_conv(&m->stem) || !m->fc_weights || !m->fc_bias) return -1;
    RedundantResnetPlan plan = redundant_resnet_plan(id);
    const int H=PART_IMAGE_H, C=PART_IMAGE_C;
    size_t image_bytes=(size_t)(plan.input.hi-plan.input.lo)*H*C;
    if (bytes != image_bytes) return -1;
    if (us) memset(us,0,30*sizeof(*us));
    /* Normalize exactly once before timing the first layer. */
    for (size_t i=0; i<image_bytes; ++i) g_input[i]=(float)pixels[i]/255.0f;

    int64_t tick=esp_timer_get_time();
    if (stem_from_float(g_input,plan.input,plan.stem,H,C,&m->stem)!=0)return -1;
    if (us)us[0]=(uint32_t)(esp_timer_get_time()-tick);
    Stripe current={g_feature,plan.stem.lo,plan.stem.hi,H,16};
    for (int b=0;b<9;b++) {
        const int out_w=H>>(b/3);
        uint32_t *t1=us?&us[1+3*b]:NULL;
        uint32_t *t2=us?&us[2+3*b]:NULL;
        uint32_t *ta=us?&us[3+3*b]:NULL;
        if (process_block(&current,&m->blocks[b],plan.mid[b],plan.output[b],
                          out_w,t1,t2,ta)!=0)return -1;
    }
    PartRange owned=part_range(PART_RESNET[2][id],PART_RESNET[2][id+1]);
    if (owned.hi <= owned.lo || owned.lo < current.lo ||
        owned.hi > current.hi || current.channels != 64) return -1;
    tick=esp_timer_get_time();
    float avg[64]={0};
    int count=(owned.hi-owned.lo)*current.w;
    for (int y=owned.lo; y<owned.hi; y++) {
        const float *row=local_row(&current,y);
        for (int x=0; x<current.w; x++)
            for (int c=0; c<64; c++) avg[c]+=row[(size_t)x*64+c];
    }
    for (int c=0;c<64;c++) avg[c]/=(float)count;
    if (us)us[28]=(uint32_t)(esp_timer_get_time()-tick);
    tick=esp_timer_get_time();
    for (int o=0;o<10;o++) {
        float sum=m->fc_bias[o];
        for (int c=0;c<64;c++)sum+=avg[c]*m->fc_weights[(size_t)o*64+c];
        logits[o]=sum;
    }
    if (us)us[29]=(uint32_t)(esp_timer_get_time()-tick);
    return 0;
}

int resnet_infer_local(const uint8_t *p,size_t n,int id,
                      const ResnetWeights *m,float logits[10]) {
    return resnet_infer_local_timed(p,n,id,m,logits,NULL);
}

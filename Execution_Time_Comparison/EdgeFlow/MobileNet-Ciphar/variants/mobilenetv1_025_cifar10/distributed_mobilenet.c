/* Exact globally aligned MobileNetV1-0.25 spatial stripes, with FP32 halos.
 * The source/target boundaries for ALL stages are in partition_config.h.
 * Depthwise kernels use interlayer source halo; pointwise is row-local.
 * When H becomes 1 only one ESP32 owns the last feature row.
 * The other devices have zero work for GAP/FC, not independent predictions.
 */
#include "distributed_mobilenet.h"
#include "partition_config.h"
#include "halo_exchange.h"
#include "esp_timer.h"
#include <string.h>

#define SCRATCH_FLOATS 6144
static float work[2][SCRATCH_FLOATS];

static int valid(const MobileConv *c) {
    return c && c->weights && c->bias && c->in_channels && c->out_channels &&
           (c->kernel == 1 || c->kernel == 3) &&
           (c->stride == 1 || c->stride == 2) &&
           (!c->depthwise || (c->kernel==3 && c->in_channels==c->out_channels));
}
/* output starts/ends globally; input is compact source stripe. */
static int apply(const float *src, float *dst, int input_h, int input_w,
                 int first, int last, const MobileConv *c,
                 int out_first, int out_last)
{
    if (!valid(c) || !src || !dst) return -1;
    const int oh=(input_h+c->stride-1)/c->stride;
    const int ow=(input_w+c->stride-1)/c->stride;
    const int ic=c->in_channels, oc=c->out_channels;
    if (out_first<0 || out_last>oh || out_first>out_last ||
        (size_t)(out_last-out_first)*ow*oc>SCRATCH_FLOATS) return -1;
    int pad=c->kernel==3?1:0;
    for (int oy=out_first;oy<out_last;oy++)
        for (int ox=0;ox<ow;ox++)
            for (int o=0;o<oc;o++) {
                float sum=c->bias[o];
                for (int ky=0;ky<c->kernel;ky++) {
                    int iy=oy*c->stride+ky-pad;
                    if (iy<0 || iy>=input_h) continue;
                    const float *row=halo_row(src,iy,first,last,input_w,ic);
                    if (!row) return -1;
                    for (int kx=0;kx<c->kernel;kx++) {
                        int ix=ox*c->stride+kx-pad;
                        if (ix<0 || ix>=input_w) continue;
                        const float *x=row+(size_t)ix*ic;
                        if (c->depthwise) {
                            size_t wi=((size_t)o*3+ky)*3+kx;
                            sum+=x[o]*c->weights[wi];
                        } else {
                            const float *weight=c->weights+
                                (size_t)o*ic*c->kernel*c->kernel+
                                (size_t)ky*c->kernel+kx;
                            for (int ci=0;ci<ic;ci++)
                                sum+=x[ci]*weight[(size_t)ci*c->kernel*c->kernel];
                        }
                    }
                }
                if (sum<0.0f) sum=0.0f;
                dst[((size_t)(oy-out_first)*ow+ox)*oc+o]=sum;
            }
    return 0;
}
int mobilenet_infer_local_timed(const uint8_t *pixels, size_t bytes,
             int device_id, const MobileWeights *model, float logits[10], uint32_t *us)
{
    if (!pixels || !model || !logits || device_id<0 || device_id>2 ||
        !model->fc_weights || !model->fc_bias || !valid(&model->stem) ||
        model->stem.in_channels!=PART_IMAGE_C ||
        model->stem.out_channels!=8 || model->stem.kernel!=3 ||
        model->stem.stride!=2 || model->stem.depthwise) return -1;
    if (us) memset(us,0,29*sizeof(uint32_t));
    int first=PART_MOBILE[0][device_id],last=PART_MOBILE[0][device_id+1];
    int h=PART_IMAGE_H,w=PART_IMAGE_W, channels=PART_IMAGE_C, stage=0;
    size_t n=(size_t)(last-first)*w*channels;
    if (bytes!=n || n>SCRATCH_FLOATS) return -1;
    for(size_t i=0;i<n;i++) work[0][i]=(float)pixels[i]/255.0f;
    int active=0;
    if (halo_prepare(&(HaloSpec){.step=0,.source_height=h,.width=w,
          .channels=channels,.kernel=3,.stride=2,.pad=1,
          .source_bounds=PART_MOBILE[0],.destination_bounds=PART_MOBILE[1],
          .src=work[active],.compact=1})!=0) return -1;
    int next_first=PART_MOBILE[1][device_id],next_last=PART_MOBILE[1][device_id+1];
    int64_t tick=esp_timer_get_time();
    if (apply(work[active],work[1-active],h,w,first,last,&model->stem,
              next_first,next_last)!=0) return -1;
    if (us) us[0]=next_last>next_first?(uint32_t)(esp_timer_get_time()-tick):0;
    active=1-active; h=(h+1)/2; w=(w+1)/2;
    first=next_first; last=next_last; channels=8; stage=1;
    for (int i=0;i<MOBILE_BLOCKS;i++) {
        const MobileBlock *b=&model->blocks[i];
        const MobileConv *d=&b->depthwise, *p=&b->pointwise;
        if (!valid(d) || !valid(p) || !d->depthwise || p->depthwise ||
            d->in_channels!=channels || d->out_channels!=channels ||
            d->kernel!=3 || p->kernel!=1 || p->stride!=1 ||
            p->in_channels!=channels) return -1;
        int next_stage=stage+(d->stride==2?1:0);
        if(next_stage>5) return -1;
        const uint8_t *next_bounds=PART_MOBILE[next_stage];
        next_first=next_bounds[device_id];next_last=next_bounds[device_id+1];
        if (halo_prepare(&(HaloSpec){.step=1+2*i,.source_height=h,.width=w,
              .channels=channels,.kernel=3,.stride=d->stride,.pad=1,
              .source_bounds=PART_MOBILE[stage],.destination_bounds=next_bounds,
              .src=work[active],.compact=1})!=0) return -1;
        tick=esp_timer_get_time();
        if (apply(work[active],work[1-active],h,w,first,last,d,
                  next_first,next_last)!=0) return -1;
        if (us) us[1+2*i]=next_last>next_first?
                          (uint32_t)(esp_timer_get_time()-tick):0;
        active=1-active;h=(h+d->stride-1)/d->stride;
        w=(w+d->stride-1)/d->stride;
        first=next_first;last=next_last;stage=next_stage;
        tick=esp_timer_get_time();
        if (apply(work[active],work[1-active],h,w,first,last,p,
                  first,last)!=0) return -1;
        if (us) us[2+2*i]=last>first?(uint32_t)(esp_timer_get_time()-tick):0;
        active=1-active;channels=p->out_channels;
    }
    if (h!=1 || w!=1 || channels!=256) return -1;
    if (last==first) {
        memset(logits,0,10*sizeof(float));
        return 0; /* no owned final position for this device */
    }
    tick=esp_timer_get_time();
    float mean[256]={0};
    const int count=(last-first)*w;
    for(int pos=0;pos<count;pos++)
        for(int c=0;c<256;c++) mean[c]+=work[active][(size_t)pos*256+c];
    for(int c=0;c<256;c++) mean[c]/=(float)count;
    if (us) us[27]=(uint32_t)(esp_timer_get_time()-tick);
    tick=esp_timer_get_time();
    for(int cls=0;cls<10;cls++) {
        float sum=model->fc_bias[cls];
        for(int c=0;c<256;c++) sum+=mean[c]*model->fc_weights[(size_t)cls*256+c];
        logits[cls]=sum;
    }
    if (us) us[28]=(uint32_t)(esp_timer_get_time()-tick);
    return 0;
}
int mobilenet_infer_local(const uint8_t *pixels,size_t bytes,int device_id,
                          const MobileWeights *model,float logits[10])
{
    return mobilenet_infer_local_timed(pixels,bytes,device_id,model,logits,NULL);
}

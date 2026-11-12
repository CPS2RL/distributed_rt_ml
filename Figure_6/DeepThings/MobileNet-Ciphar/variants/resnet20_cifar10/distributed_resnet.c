/* ResNet-20: communication-free redundant row execution.
 * Exact backward receptive-field expansion from each device's final rows.
 * No intermediate ESP-NOW or halo lookups: all required rows are local.
 */
#include "distributed_resnet.h"
#include "partition_config.h"
#include <string.h>
#include "esp_timer.h"

#if EXP_IMG_W == 28
#define MAX_ACT_FLOATS 12544
#define PROJ_FLOATS 6272
#elif EXP_IMG_W == 32
#define MAX_ACT_FLOATS 16384
#define PROJ_FLOATS 8192
#else
#error ResNet-20 expects 28 or 32
#endif
static float g_work[3][MAX_ACT_FLOATS];
static float g_projection[PROJ_FLOATS];

typedef struct {
    const float *data;
    int lo, hi, w, channels;
} Stripe;

static int valid_conv(const ResnetConv *c) {
    return c && c->weights && c->bias && c->input_channels > 0 &&
           c->output_channels > 0 && (c->kernel == 1 || c->kernel == 3) &&
           (c->stride == 1 || c->stride == 2);
}
static const float *local_row(const Stripe *in, int r) {
    if (r < in->lo || r >= in->hi) return NULL;
    return in->data + (size_t)(r-in->lo)*in->w*in->channels;
}
static int conv(const Stripe *in, const ResnetConv *c,
                int out_w, PartRange rows, int relu, float *dst) {
    int ci=c->input_channels, co=c->output_channels;
    int k=c->kernel, stride=c->stride, pad=k/2;
    if ((size_t)(rows.hi-rows.lo)*out_w*co>MAX_ACT_FLOATS) return -1;
    for (int oy=rows.lo;oy<rows.hi;oy++)
        for (int ox=0;ox<out_w;ox++)
            for (int oc=0;oc<co;oc++) {
                float sum=c->bias[oc];
                for (int ky=0;ky<k;ky++) {
                    int iy=oy*stride + ky-pad;
                    if (iy<0 || iy >= in->w) continue;
                    const float *row=local_row(in,iy);
                    if (!row) return -1; /* Missing dependency is an error. */
                    for (int kx=0;kx<k;kx++) {
                        int ix=ox*stride+kx-pad;
                        if (ix<0 || ix>=in->w) continue;
                        const float *x=row+(size_t)ix*ci;
                        for (int ic=0;ic<ci;ic++) {
                            size_t wi=(((size_t)oc*ci+ic)*k+ky)*k+kx;
                            sum+=x[ic]*c->weights[wi];
                        }
                    }
                }
                if (relu && sum<0.0f) sum=0.0f;
                dst[((size_t)(oy-rows.lo)*out_w+ox)*co+oc]=sum;
            }
    return 0;
}
static int project(const Stripe *in, const ResnetConv *c,
                   int out_w, PartRange r, float *dst) {
    int ci=c->input_channels, co=c->output_channels;
    if((size_t)(r.hi-r.lo)*out_w*co>PROJ_FLOATS) return -1;
    for(int oy=r.lo;oy<r.hi;oy++)
        for(int ox=0;ox<out_w;ox++) {
            const float *row=local_row(in,oy*c->stride);
            if (!row) return -1;
            const float *x=row+(size_t)(ox*c->stride)*ci;
            for(int oc=0;oc<co;oc++) {
                float val=c->bias[oc];
                for(int ic=0;ic<ci;ic++) val+=x[ic]*c->weights[(size_t)oc*ci+ic];
                dst[((size_t)(oy-r.lo)*out_w+ox)*co+oc]=val;
            }
        }
    return 0;
}
int resnet_input_rows(int id,int *lo,int *hi) {
    if(id<0||id>=3||!lo||!hi)return -1;
    RedundantResnetPlan p=redundant_resnet_plan(id);
    *lo=p.input.lo; *hi=p.input.hi; return 0;
}
int resnet_infer_local_timed(const uint8_t *pixels,size_t bytes,int id,
        const ResnetWeights *m,float logits[10],uint32_t *us) {
    if(!pixels||!m||!logits||id<0||id>2||!valid_conv(&m->stem)||
       !m->fc_weights||!m->fc_bias)return -1;
    RedundantResnetPlan plan=redundant_resnet_plan(id);
    const int H=PART_IMAGE_H,C=PART_IMAGE_C;
    size_t n=(size_t)(plan.input.hi-plan.input.lo)*H*C;
    if(bytes!=n||n>MAX_ACT_FLOATS)return -1;
    if(us)memset(us,0,30*sizeof(*us));
    for(size_t i=0;i<n;i++) g_work[0][i]=(float)pixels[i]/255.0f;
    Stripe in={g_work[0],plan.input.lo,plan.input.hi,H,C};
    int active=1;
    int64_t tick=esp_timer_get_time();
    if(conv(&in,&m->stem,H,plan.stem,1,g_work[active])!=0)return -1;
    if(us)us[0]=(uint32_t)(esp_timer_get_time()-tick);
    in=(Stripe){g_work[active],plan.stem.lo,plan.stem.hi,H,16};
    for(int b=0;b<9;b++) {
        const ResnetBlock *blk=&m->blocks[b];
        if(!valid_conv(&blk->conv1)||!valid_conv(&blk->conv2)||
           blk->conv1.input_channels!=in.channels)return -1;
        int w=H>>(b/3), ch=blk->conv1.output_channels;
        PartRange mid=plan.mid[b], out=plan.output[b];
        int t1=(active+1)%3,t2=(active+2)%3;
        uint32_t projection_us=0;
        if(blk->projection.weights) {
            if(!valid_conv(&blk->projection))return -1;
            tick=esp_timer_get_time();
            if(project(&in,&blk->projection,w,out,g_projection)!=0)return -1;
            projection_us=(uint32_t)(esp_timer_get_time()-tick);
        }
        tick=esp_timer_get_time();
        if(conv(&in,&blk->conv1,w,mid,1,g_work[t1])!=0)return -1;
        if(us)us[1+3*b]=(uint32_t)(esp_timer_get_time()-tick);
        Stripe middle={g_work[t1],mid.lo,mid.hi,w,ch};
        tick=esp_timer_get_time();
        if(conv(&middle,&blk->conv2,w,out,0,g_work[t2])!=0)return -1;
        if(us)us[2+3*b]=(uint32_t)(esp_timer_get_time()-tick);
        tick=esp_timer_get_time();
        for(int oy=out.lo;oy<out.hi;oy++)
            for(int ox=0;ox<w;ox++)
                for(int oc=0;oc<ch;oc++) {
                    size_t oi=((size_t)(oy-out.lo)*w+ox)*ch+oc;
                    float skip;
                    if(blk->projection.weights) skip=g_projection[oi];
                    else {
                        const float *row=local_row(&in,oy);
                        if(!row) return -1;
                        skip=row[(size_t)ox*ch+oc];
                    }
                    float sum=g_work[t2][oi]+skip;
                    g_work[t1][oi]=sum>0.0f?sum:0.0f;
                }
        if(us)us[3+3*b]=(uint32_t)(esp_timer_get_time()-tick)+projection_us;
        active=t1;
        in=(Stripe){g_work[active],out.lo,out.hi,w,ch};
    }
    PartRange owned=part_range(PART_RESNET[2][id],PART_RESNET[2][id+1]);
    if(owned.hi<=owned.lo||owned.lo<in.lo||owned.hi>in.hi||in.channels!=64)return -1;
    tick=esp_timer_get_time();
    float avg[64]={0};
    int count=(owned.hi-owned.lo)*in.w;
    for(int y=owned.lo;y<owned.hi;y++) {
        const float *row=local_row(&in,y);
        for(int x=0;x<in.w;x++)
            for(int c=0;c<64;c++)avg[c]+=row[(size_t)x*64+c];
    }
    for(int c=0;c<64;c++)avg[c]/=(float)count;
    if(us)us[28]=(uint32_t)(esp_timer_get_time()-tick);
    tick=esp_timer_get_time();
    for(int o=0;o<10;o++) {
        float sum=m->fc_bias[o];
        for(int c=0;c<64;c++)sum+=avg[c]*m->fc_weights[(size_t)o*64+c];
        logits[o]=sum;
    }
    if(us)us[29]=(uint32_t)(esp_timer_get_time()-tick);
    return 0;
}
int resnet_infer_local(const uint8_t *p,size_t n,int id,
                      const ResnetWeights *m,float logits[10]) {
    return resnet_infer_local_timed(p,n,id,m,logits,NULL);
}

#ifndef PARTITION_CONFIG_H
#define PARTITION_CONFIG_H
/* THE ONE FILE TO CHANGE FOR ALL LAYER BOUNDARIES.
 * Four values per layer: [0, end_device0, end_device1, H].
 * All ranges are [begin,end), globally indexed.
 * Every conv/pool follows the SAME global spatial tensor layout.
 * Changing a boundary here updates input distribution AND all layer work.
 * A zero-row partition is valid (MobileNet downsampling may force it).
 */
#include <stdint.h>

#if EXP_DATASET_ID == 1
#define PART_IMAGE_H 28
#define PART_IMAGE_W 28
#define PART_IMAGE_C 1
static const uint8_t PART_LENET[5][4] = {
    {0, 9, 18, 28},   /* 28 input */
    {0, 8, 16, 24},   /* 24 Conv1 */
    {0, 4,  8, 12},   /* 12 Pool1 */
    {0, 2,  4,  8},   /* 8 Conv2 */
    {0, 1,  2,  4}    /* 4 Pool2 */
};
static const uint8_t PART_RESNET[3][4] = {
    {0, 9, 18, 28},   /* Stem & stage1: 28 */
    {0, 5,  9, 14},   /* Stage2: 14 */
    {0, 3,  5,  7}    /* Stage3: 7 */
};
static const uint8_t PART_MOBILE[6][4] = {
    {0, 9, 18, 28},   /* input:28 */
    {0, 5,  9, 14},   /* stem, D1:14 */
    {0, 3,  5,  7},   /* D2,D3:7 */
    {0, 2,  3,  4},   /* D4,D5:4 */
    {0, 1,  2,  2},   /* D6..D11:2 */
    {0, 1,  1,  1}    /* D12,D13:1, only device0 owns last row */
};
#else
#define PART_IMAGE_H 32
#define PART_IMAGE_W 32
#define PART_IMAGE_C 3
static const uint8_t PART_LENET[5][4] = {
    {0, 11, 22, 32}, /* 32 input */
    {0, 10, 20, 28}, /* 28 Conv1 */
    {0,  5, 10, 14}, /* 14 Pool1 */
    {0,  4,  8, 10}, /* 10 Conv2 */
    {0,  2,  4,  5}  /* 5 Pool2 */
};
static const uint8_t PART_RESNET[3][4] = {
    {0, 11, 22, 32}, /* Stem & stage1:32 */
    {0,  6, 11, 16}, /* Stage2:16 */
    {0,  3,  5,  8}  /* Stage3:8 */
};
static const uint8_t PART_MOBILE[6][4] = {
    {0, 11, 22, 32}, /* input:32 */
    {0,  6, 11, 16}, /* stem, D1:16 */
    {0,  3,  5,  8}, /* D2,D3:8 */
    {0,  2,  3,  4}, /* D4,D5:4 */
    {0,  1,  2,  2}, /* D6..D11:2 */
    {0,  1,  1,  1}  /* D12,D13:1 */
};
#endif

#if EXP_MODEL_ID == 1
#define PART_SELECTED_INPUT PART_LENET[0]
#elif EXP_MODEL_ID == 2
#define PART_SELECTED_INPUT PART_RESNET[0]
#else
#define PART_SELECTED_INPUT PART_MOBILE[0]
#endif

/* Reject non-monotonic, height-mismatched or empty INPUT partitions at runtime.
 * Intermediate zero-row device partitions are allowed in MobileNet.
 * A halo-exchange operation detects missing rows at runtime and fails rather
 * than silently replacing them with zero.
 */
static inline int partition_valid(const uint8_t p[4], int h, int input)
{
    if (p[0] != 0 || p[3] != h) return 0;
    for (int i = 0; i < 3; ++i)
        if (p[i] > p[i+1] || (input && p[i] == p[i+1])) return 0;
    return 1;
}
static inline int all_partitions_valid(void)
{
    static const int lenet_h_mnist[5] = {28,24,12,8,4};
    static const int lenet_h_cifar[5] = {32,28,14,10,5};
    static const int res_h_mnist[3] = {28,14,7};
    static const int res_h_cifar[3] = {32,16,8};
    static const int mob_h_mnist[6] = {28,14,7,4,2,1};
    static const int mob_h_cifar[6] = {32,16,8,4,2,1};
    const int *lh = PART_IMAGE_H==28 ? lenet_h_mnist : lenet_h_cifar;
    const int *rh = PART_IMAGE_H==28 ? res_h_mnist : res_h_cifar;
    const int *mh = PART_IMAGE_H==28 ? mob_h_mnist : mob_h_cifar;
    for (int i=0;i<5;i++) if (!partition_valid(PART_LENET[i],lh[i],i==0)) return 0;
    for (int i=0;i<3;i++) if (!partition_valid(PART_RESNET[i],rh[i],i==0)) return 0;
    for (int i=0;i<6;i++) if (!partition_valid(PART_MOBILE[i],mh[i],i==0)) return 0;
    return 1;
}


/* ================== COMMUNICATION-FREE REDUNDANT PLANS ==================
 * A plan gives the exact extra input rows each board must compute locally
 * to reproduce its OWNED final output rows without exchanging activations.
 * Half-open global row coordinates, independent of device crop origin.
 */
typedef struct { int lo, hi; } PartRange;
static inline PartRange part_range(int lo, int hi) {
    PartRange x = {lo, hi}; return x;
}
static inline PartRange part_dependency(PartRange out, int source_h,
                                        int stride, int kernel, int pad) {
    if (out.hi <= out.lo) return part_range(0, 0);
    int lo=out.lo * stride - pad;
    int hi=(out.hi - 1) * stride - pad + kernel;
    if (lo < 0) lo=0;
    if (hi > source_h) hi=source_h;
    return part_range(lo, hi);
}
/* For LeNet-5: input -> C1 -> P1 -> C2 -> P2.
 * Final Pool2 rows are owned disjointly; preceding stages expand backwards.
 */
static inline void redundant_lenet_plan(int id, PartRange spans[5])
{
    const int H = PART_IMAGE_H;

    spans[4] = part_range(
        PART_LENET[4][id],
        PART_LENET[4][id+1]
    );

    spans[3] = part_dependency(spans[4], (H-12)/2, 2, 2, 0);
    spans[2] = part_dependency(spans[3], (H-4)/2, 1, 5, 0);

    /* Full Conv1 feature map on every device */
    spans[1] = part_range(0, H-4);

    /* Full input image on every device */
    spans[0] = part_range(0, H);
}
typedef struct {
    PartRange input, stem;
    PartRange output[9], mid[9];
} RedundantResnetPlan;
static inline RedundantResnetPlan redundant_resnet_plan(int id) {
    RedundantResnetPlan p;
    const int H=PART_IMAGE_H;
    PartRange need=part_range(PART_RESNET[2][id], PART_RESNET[2][id+1]);
    for(int b=8;b>=0;--b) {
        int stage=b/3;
        int output_h=H>>stage;
        int input_h=(b==3 || b==6) ? output_h*2 : output_h;
        int stride=(b==3 || b==6) ? 2 : 1;
        p.output[b]=need;
        p.mid[b]=part_dependency(need, output_h, 1, 3, 1);
        need=part_dependency(p.mid[b],input_h,stride,3,1);
        /* The 1x1 projection/identity skip is contained in this span. */
    }
    p.stem=need;
    p.input=part_range(0,H);
    return p;
}
/* MobileNetV1-0.25 downsamples to one spatial row. Three disjoint final
 * output stripes are impossible, and the receptive field covers the image.
 * Use complete pre-input replication: three real independent full-image
 * executions, each doing redundant local work without intermediate radio.
 */
static inline PartRange redundant_input_span(int id)
{
    (void)id;
    return part_range(0, PART_IMAGE_H);
}
#endif

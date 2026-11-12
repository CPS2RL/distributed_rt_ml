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
#endif

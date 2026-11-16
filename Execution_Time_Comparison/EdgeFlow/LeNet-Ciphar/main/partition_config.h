
#ifndef PARTITION_CONFIG_H
#define PARTITION_CONFIG_H


#include <stdint.h>

/* =========================================================
 * MNIST
 * ========================================================= */

#if EXP_DATASET_ID == 1

#define PART_IMAGE_H 28
#define PART_IMAGE_W 28
#define PART_IMAGE_C 1

/* ---------------------------------------------------------
 * LeNet-5 MNIST
 *
 * Input:  28 rows
 * Conv1:  24 rows
 * Pool1:  12 rows
 * Conv2:   8 rows
 * Pool2:   4 rows
 * --------------------------------------------------------- */

static const uint8_t PART_LENET[5][4] = {

    /* Input: ESP1=13, ESP2=8, ESP3=7 */
    {0, 16, 21, 28},

    /* Conv1: ESP1=13, ESP2=7, ESP3=4 */
    {0, 16, 20, 24},

    /* Pool1: ESP1=7, ESP2=3, ESP3=2 */
    {0, 8, 10, 12},

    /* Conv2: ESP1=5, ESP2=2, ESP3=1 */
    {0, 5, 7, 8},

    /* Pool2: ESP1=2, ESP2=1, ESP3=1 */
    {0, 2, 3, 4}
};


/* ---------------------------------------------------------
 * ResNet-20 MNIST
 *
 * Stem + Stage1: 28 rows
 * Stage2:        14 rows
 * Stage3:         7 rows
 * --------------------------------------------------------- */

static const uint8_t PART_RESNET[3][4] = {

    /* Stem + Stage1: ESP1=13, ESP2=8, ESP3=7 */
    {0, 16, 21, 28},

    /* Stage2: ESP1=9, ESP2=3, ESP3=2 */
    {0, 10, 12, 14},

    /* Stage3: ESP1=5, ESP2=1, ESP3=1 */
    {0, 5, 6, 7}
};


/* ---------------------------------------------------------
 * MobileNetV1-0.25 MNIST
 *
 * Input:       28 rows
 * Stem + D1:   14 rows
 * D2-D3:        7 rows
 * D4-D5:        4 rows
 * D6-D11:       2 rows
 * D12-D13:      1 row
 * --------------------------------------------------------- */

static const uint8_t PART_MOBILE[6][4] = {

    /* Input: ESP1=13, ESP2=8, ESP3=7 */
    {0, 16, 21, 28},

    /* Stem + D1: ESP1=9, ESP2=3, ESP3=2 */
    {0, 10, 12, 14},

    /* D2-D3: ESP1=5, ESP2=1, ESP3=1 */
    {0, 5, 6, 7},

    /* D4-D5: ESP1=2, ESP2=1, ESP3=1 */
    {0, 2, 3, 4},

    /* D6-D11: ESP1=1, ESP2=1, ESP3=0 */
    {0, 1, 2, 2},

    /* D12-D13: ESP1=1, ESP2=0, ESP3=0 */
    {0, 1, 1, 1}
};


/* =========================================================
 * CIFAR-10
 * ========================================================= */

#else

#define PART_IMAGE_H 32
#define PART_IMAGE_W 32
#define PART_IMAGE_C 3

/* ---------------------------------------------------------
 * LeNet-5 CIFAR-10
 *
 * Input:  32 rows
 * Conv1:  28 rows
 * Pool1:  14 rows
 * Conv2:  10 rows
 * Pool2:   5 rows
 * --------------------------------------------------------- */

static const uint8_t PART_LENET[5][4] = {

    /* Input: ESP1=12, ESP2=11, ESP3=9 */
    {0, 14, 25, 32},

    /* Conv1: ESP1=12, ESP2=10, ESP3=6 */
    {0, 13, 20, 28},

    /* Pool1: ESP1=8, ESP2=4, ESP3=2 */
    {0, 9, 12, 14},

    /* Conv2: ESP1=6, ESP2=3, ESP3=1 */
    {0, 4, 7, 10},

    /* Pool2: ESP1=3, ESP2=1, ESP3=1 */
    {0, 2, 4, 5}
};


/* ---------------------------------------------------------
 * ResNet-20 CIFAR-10
 *
 * Stem + Stage1: 32 rows
 * Stage2:        16 rows
 * Stage3:         8 rows
 * --------------------------------------------------------- */

static const uint8_t PART_RESNET[3][4] = {

    /* Stem + Stage1: ESP1=12, ESP2=11, ESP3=9 */
    {0, 15, 25, 32},

    /* Stage2: ESP1=10, ESP2=4, ESP3=2 */
    {0, 10, 14, 16},

    /* Stage3: ESP1=6, ESP2=1, ESP3=1 */
    {0, 6, 7, 8}
};


/* ---------------------------------------------------------
 * MobileNetV1-0.25 CIFAR-10
 *
 * Input:       32 rows
 * Stem + D1:   16 rows
 * D2-D3:        8 rows
 * D4-D5:        4 rows
 * D6-D11:       2 rows
 * D12-D13:      1 row
 * --------------------------------------------------------- */

static const uint8_t PART_MOBILE[6][4] = {

    /* Input: ESP1=12, ESP2=11, ESP3=9 */
    {0, 15, 25, 32},

    /* Stem + D1: ESP1=10, ESP2=4, ESP3=2 */
    {0, 10, 14, 16},

    /* D2-D3: ESP1=6, ESP2=1, ESP3=1 */
    {0, 6, 7, 8},

    /* D4-D5: ESP1=2, ESP2=1, ESP3=1 */
    {0, 2, 3, 4},

    /* D6-D11: ESP1=1, ESP2=1, ESP3=0 */
    {0, 1, 2, 2},

    /* D12-D13: ESP1=1, ESP2=0, ESP3=0 */
    {0, 1, 1, 1}
};

#endif


/* =========================================================
 * SELECTED MODEL INPUT PARTITION
 * ========================================================= */

#if EXP_MODEL_ID == 1

#define PART_SELECTED_INPUT PART_LENET[0]

#elif EXP_MODEL_ID == 2

#define PART_SELECTED_INPUT PART_RESNET[0]

#else

#define PART_SELECTED_INPUT PART_MOBILE[0]

#endif


/* =========================================================
 * PARTITION VALIDATION
 * ========================================================= */

/*
 * Check:
 *
 * 1. First boundary must be zero.
 * 2. Last boundary must equal layer height.
 * 3. Boundaries must be nondecreasing.
 * 4. Input partitions must be nonempty.
 *
 * Intermediate zero-row partitions are allowed.
 */

static inline int partition_valid(
    const uint8_t p[4],
    int h,
    int input
)
{
    if (p[0] != 0 || p[3] != h)
        return 0;

    for (int i = 0; i < 3; ++i)
    {
        if (p[i] > p[i + 1])
            return 0;

        if (input && p[i] == p[i + 1])
            return 0;
    }

    return 1;
}


/* =========================================================
 * VALIDATE ALL MODEL PARTITIONS
 * ========================================================= */

static inline int all_partitions_valid(void)
{
    /* LeNet-5 layer heights */

    static const int lenet_h_mnist[5] = {
        28, 24, 12, 8, 4
    };

    static const int lenet_h_cifar[5] = {
        32, 28, 14, 10, 5
    };

    /* ResNet-20 stage heights */

    static const int res_h_mnist[3] = {
        28, 14, 7
    };

    static const int res_h_cifar[3] = {
        32, 16, 8
    };

    /* MobileNetV1-0.25 stage heights */

    static const int mob_h_mnist[6] = {
        28, 14, 7, 4, 2, 1
    };

    static const int mob_h_cifar[6] = {
        32, 16, 8, 4, 2, 1
    };

    /* Select heights based on dataset */

    const int *lh =
        PART_IMAGE_H == 28
            ? lenet_h_mnist
            : lenet_h_cifar;

    const int *rh =
        PART_IMAGE_H == 28
            ? res_h_mnist
            : res_h_cifar;

    const int *mh =
        PART_IMAGE_H == 28
            ? mob_h_mnist
            : mob_h_cifar;


    /* Validate LeNet-5 */

    for (int i = 0; i < 5; ++i)
    {
        if (!partition_valid(
                PART_LENET[i],
                lh[i],
                i == 0))
        {
            return 0;
        }
    }


    /* Validate ResNet-20 */

    for (int i = 0; i < 3; ++i)
    {
        if (!partition_valid(
                PART_RESNET[i],
                rh[i],
                i == 0))
        {
            return 0;
        }
    }


    /* Validate MobileNetV1-0.25 */

    for (int i = 0; i < 6; ++i)
    {
        if (!partition_valid(
                PART_MOBILE[i],
                mh[i],
                i == 0))
        {
            return 0;
        }
    }

    return 1;
}

#endif /* PARTITION_CONFIG_H */

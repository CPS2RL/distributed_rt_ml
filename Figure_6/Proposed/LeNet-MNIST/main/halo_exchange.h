#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H
#include <stdint.h>
/* Called BEFORE any spatial operation that can require remote source rows.
 * Source HWC tensor occupies own rows [source_bounds[id],source_bounds[id+1]);
 * compact=1: stored at src[0]; compact=0: stored at src[global row*width*c].
 * Output rows are defined by destination_bounds.
 * kernel,stride,pad define vertical receptive-field geometry.
 * step is the unique operation index in the layer-time vector.
 * Returns 0 on successful, ACKed exchange or nonzero on protocol failure.
 */
typedef struct {
    int step, source_height, width, channels, kernel, stride, pad;
    const uint8_t *source_bounds, *destination_bounds;
    float *src;
    int compact;
} HaloSpec;
int halo_prepare(const HaloSpec *s);
/* The convolution can read a globally indexed input row using this lookup.
 * Returns NULL when outside input image; a requested missing row is an error.
 */
const float *halo_row(const float *local, int global_row,
                      int first, int last, int width, int channels);
/* Diagnostics provided by coordinator; halo_us[step] is radio+barrier time. */
uint32_t halo_step_us(int step);
void halo_begin_run(int run_id);
#endif

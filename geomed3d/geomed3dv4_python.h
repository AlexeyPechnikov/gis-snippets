#ifndef GEOMED_PYTHON_H
#define GEOMED_PYTHON_H

#ifdef __cplusplus
extern "C" {
#endif

const char* pygeomed_stat(uint32_t x);
uint32_t pygeomed_stats(void);

void pygeomed(const GEOMED_DATAIN *image_raw, uint32_t image_width, uint32_t image_height,
            int32_t *mask_x, int32_t *mask_y, int32_t *mask_z, uint32_t mask_length,
            uint32_t option_min, uint32_t option_max, GEOMED_DATAIN option_nodata,
            GEOMED_DATAOUT* statsall);

#ifdef __cplusplus
}
#endif

#endif  // GEOMED_PYTHON_H

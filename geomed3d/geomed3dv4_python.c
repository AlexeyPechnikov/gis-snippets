
const char* pygeomed_stat(uint32_t stat){
    return GEOMED_STAT_NAMES[stat];
}

uint32_t pygeomed_stats(){
    return GEOMED_STATS;
}

void pygeomed(const GEOMED_DATAIN *image_raw, uint32_t image_width, uint32_t image_height,
            int32_t *mask_x, int32_t *mask_y, int32_t *mask_z, uint32_t mask_length,
            uint32_t option_min, uint32_t option_max, GEOMED_DATAIN option_nodata,
            GEOMED_DATAOUT* statsall) {

    geomed_option option;
    option.min    = option_min;
    option.max    = option_max;
    option.nodata = option_nodata;

    geomed_image image;
    image.raw    = image_raw;
    image.width  = image_width;
    image.height = image_height;

    geomed_mask mask;
    mask.x      = mask_x;
    mask.y      = mask_y;
    mask.z      = mask_z;
    mask.length = mask_length;

    geomed(image, mask, option, statsall);

    return;
}


GEOMED_DATAOUT *geomed_stat(geomed_image *image, geomed_filter *filters, size_t image_x, size_t image_y,
        const geomed_option *option) {
    // use "long" datatype for linear index to prevent compiler related issues
    size_t idx, curr_idx, r, size_r; // linear indicies
    long curr_x, curr_y; // decart indicies
    GEOMED_DATAOUT *stats;
    const geomed_filter *filter;
    // итератор по пикселам
    GEOMED_DATAIN curr;
    long count;
    // circle calculation variables
    GEOMED_DATATYPE mean;

    // filters range
    size_r = option->max - option->min + 1;

    stats = calloc( GEOMED_STATS*size_r, sizeof(GEOMED_DATAOUT));
    if (!stats)
        handle_error("Memory allocation error on stats array initialization!");

    {
        // orig value
        if ( image_y < 0 || image_y >= image->height || image_x < 0 || image_x >= image->width) {
            // при выходе за границу изображения игнорировать пиксель
        } else {
            // calculate linear coordinate of filtering image array element
            curr_idx = image_y*image->width + image_x;
            // current pixel of image
            curr = image->raw[curr_idx];
            // ignore NoData values
            for ( r = 0; r < size_r && curr != option->nodata; r++ )
                stats[size_r*GEOMED_STAT_ORIG + r] = curr;
        }
    }

    for ( r = 0; r < size_r; r++ ) {
		filter = &filters[r];
		count = mean = 0;
		for (idx=0; idx<filter->length; idx++) {
			// если пиксель не удовлетворяет ограничениям, пропускаем
			// координата пикселя изображения
			curr_x = image_x + filter->x[idx];
			curr_y = image_y + filter->y[idx];
			// при выходе за границу изображения игнорировать пиксель
			if ( curr_y < 0 || curr_y >= image->height || curr_x < 0 || curr_x >= image->width) continue;
			// calculate linear coordinate of filtering image array element
			curr_idx = curr_y*image->width + curr_x;
			// current pixel of image
			curr = image->raw[curr_idx];

			// ignore NoData values
			if ( curr == option->nodata ) continue;
			
			mean  += 1.*curr;
			
			count++;
		}
		
		// should be ignored later
		if ( count < GEOMED_NUM ) count = 1;
		
		mean  = 1.*mean/count;
		
		// output values
		stats[size_r*GEOMED_STAT_ROTMEAN   + r] = ( count < GEOMED_NUM         ) ? NAN : mean;
    }
    
    return stats;
}

#include "geomed3dv4.h"

#include "geomed3dv4_stat.c"
#include "geomed3dv4_python.c"

// filter cleanup
#ifdef BUILDING_WIN
__stdcall
#endif
void geomed_filter_del(geomed_filter *filter){
    // memory is not allocated
    if ( filter ==  NULL ) return;
    // memory allocated
    if (filter->x) {
	free(filter->x);
	filter->x = NULL;
    }
    if (filter->y) {
	free(filter->y);
	filter->y = NULL;
    }
}
// make circle filter
// outer radius R, inner radius R-1 (including outer border and excluding inner border)
#ifdef BUILDING_WIN
__stdcall
#endif
void geomed_filter_new(geomed_filter *filter, uint32_t option_R) {
    uint32_t length, count, count0;
    //linear &  decart indicies: idx can be < 0
    int32_t x, y, idx;
    float rcurr, rmin, rmax;
    // make empty filter (0 items)
    filter->length = 0;
    filter->r = option_R;
    // check input parameter
    if ( option_R < 0 ) {
        return;
    }

    /* Allocate memory for filter */
    // filter length is approximate equal to circle length (2*PI*r < 2*4*r)
    // assert below controls that it's enough
    length = 2*4*(option_R+1)+1;
    filter->x = calloc( length, sizeof filter->x);
    filter->y = calloc( length, sizeof filter->y);
    if ( !filter->x || !filter->y )
        handle_error("Memory allocation error on filter initialization!");

    rmin = pow(option_R-1,2);
    rmax = pow(option_R,2);
    
    // radius 0 - current point only
    if ( !option_R ) {
    	filter->x[0] = 0;
    	filter->y[0] = 0;
    	filter->length = 1;
    	return;
    }

    count = 0;
    for (x=-option_R; x<=0; x++) {
        for (y=-option_R; y<=0; y++) {
            // circle
            rcurr = x*x + y*y;
            if ( rcurr>rmax || rcurr<=rmin ) continue;

            // linear index
            filter->x[count] = x;
            filter->y[count] = y;
            count++;
        }
    }

	assert(count0-2>0);

    // mirror filter -x => x
    count0 = count;
    for (idx=count0-2; idx>=0; idx--) { // x>0
    	filter->x[count] = -filter->x[idx];
		filter->y[count] =  filter->y[idx];
		count++;
    }
    // mirror filter -y => y
    count0 = count;
    for (idx=count0-2; idx>=1; idx--) { // y>0, y<R
    	filter->x[count] =  filter->x[idx];
		filter->y[count] = -filter->y[idx];
		count++;
    }
    
    filter->length = count;

    return;
}

// used separate thread for each XYZ point
#ifdef BUILDING_WIN
__stdcall
#endif
void geomed(geomed_image image, geomed_mask mask, geomed_option option, GEOMED_DATAOUT* statsall) {
    GEOMED_DATAOUT *stats;
    geomed_filter *filters, *filter;
    geomed_mask *pmask = &mask;
    // maximal z offset
    int32_t zmax, dz;
    uint32_t midx, sidx, ridx, idx, size_r, size_m;

    filters = calloc( option.max+1, sizeof(geomed_filter));
    if (!filters)
        handle_error("Memory allocation error on stats array initialization!");

#pragma omp parallel \
    shared(filters, option) \
    private(filter, ridx) \
    default(none)
{
#pragma omp for schedule(dynamic,1)
    for ( ridx = 0; /*option.min;*/ ridx <= option.max; ridx++ ) {
        // create filter for the radiusа
        filter = &filters[ridx/*-option.min*/];
        geomed_filter_new(filter, ridx);
    }
}

size_r = (option.max - option.min + 1);
size_m = pmask->length;

// calculate maximal positive z offset
zmax = 0;
if ( mask.z != NULL) {
    for (midx = 0; midx < size_m; midx++) {
        zmax = (mask.z[midx] > zmax ? mask.z[midx] : zmax);
    }
}

#pragma omp parallel \
    shared(filters, statsall, image, option, pmask,stderr) \
    private(stats, midx, sidx, ridx, idx, dz) \
    firstprivate(size_r, size_m, zmax) \
    default(none)
{
#pragma omp for schedule(static)
    for ( midx = 0; midx < size_m; midx++ ) {
        // one statistics arrays
        GEOMED_DATAOUT *stat;
        stat = calloc( size_r, sizeof(GEOMED_DATAOUT) );
        if (!stat)
            handle_error("Memory allocation error on stat array initialization!");
        if (pmask->x[midx] >=0 && pmask->y[midx] >= 0) {
            // calculate statistics [stat, r]
            stats = geomed_stat(&image, filters, pmask->x[midx], pmask->y[midx], &option);
            // apply surface z offset
            dz = zmax - pmask->z[midx];
            for ( sidx=GEOMED_STAT_ORIG; (sidx < GEOMED_STATS) & (dz != 0); sidx++) {
                idx = size_r*sidx + 0; //ridx=0
                memcpy(stat, stats + idx, size_r*sizeof(GEOMED_DATAOUT));
                for ( ridx = 0; (ridx < dz) & (ridx < size_r); ridx++ ) {
                    stats[idx + ridx] = NAN;
                }
                if (dz >= size_r) continue;
                memcpy(stats + idx + dz, stat, (size_r-dz)*sizeof(GEOMED_DATAOUT));
            }
            // transpose output array in place to [stat,r,y,x] (when input mask is 1D projection for [y,x] array)
            // and revert "r" axis
            for ( sidx=GEOMED_STAT_ORIG; sidx < GEOMED_STATS; sidx++) {
                for ( ridx = 0; ridx < size_r; ridx++ ) {
                    idx = sidx*size_r*size_m + ridx*size_m + midx;
                    statsall[idx] = stats[size_r*sidx + (size_r-ridx-1)];
                }
            }
            free(stats);
        }
        free(stat);
    }
}

    for ( ridx = 0; /*option.min;*/ ridx <= option.max; ridx++ ) {
        filter = &filters[ridx/*-option.min*/];
        geomed_filter_del(filter);
    }

    return;
}

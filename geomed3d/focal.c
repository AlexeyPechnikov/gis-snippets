#include "geomed3dv4.h"

int main(int argc, const char *argv[]){
    geomed_filter filter;
    int32_t radius, x, y;
    size_t idx;

    if( argc != 2 ) {
        fprintf(stderr, "Usage: %s radius\n", argv[0]);
        exit(1);
    }

    radius = atol( argv[1] );
    geomed_filter_new(&filter, radius);

    for (idx=0; idx<filter.length; idx++) {
        x = filter.x[idx];
        y = filter.y[idx];
        fprintf(stdout,"%d %d ", x, y );
    }

    return 0;
}

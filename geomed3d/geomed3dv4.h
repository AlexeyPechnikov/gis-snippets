#ifndef GEOMED_H
#define GEOMED_H

#include "geomed3dv4_stat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// C99
//#include <stdint.h>
#include <inttypes.h>
#include <limits.h>

#include <assert.h>

#include <omp.h>

#ifdef BUILDING_WIN
  #ifdef BUILDING_DLL
    #define DLL_DECLSPEC __declspec(dllexport) __stdcall
  #else
    #define DLL_DECLSPEC __declspec(dllimport) __stdcall
  #endif
#else
    #define DLL_DECLSPEC
#endif

// new value, min/max
#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))

// we can't calculate stats when we have no GEOMED_NUM valid pixels
// NaN value returned in this case
//http://www.gnu.org/software/libc/manual/html_node/Infinity-and-NaN.html
#define GEOMED_NUM 4

#define STREQ(a, b) (*(a) == *(b) && strcmp((a), (b)) == 0)

#define handle_error(msg) do { fprintf(stderr, msg); exit(EXIT_FAILURE); } while (0)

// internal datatype for processing
#define GEOMED_DATATYPE double
// input datatype
//#define GEOMED_DATAIN int32_t
#define GEOMED_DATAIN float
// output datatype
#define GEOMED_DATAOUT float

#ifdef __cplusplus
extern "C" {
#endif

// circle filter
typedef struct {
    uint32_t r;
    uint32_t length;
    int32_t *x;
    int32_t *y;
} geomed_filter;

// input coordinates
#define MASK_MAX_LENGTH 500*1000
typedef struct {
    uint32_t length;
    int32_t *x;
    int32_t *y;
    int32_t *z;
} geomed_mask;

// processing options
typedef struct {
    // output radius range
    uint32_t min;
    uint32_t max;
    // source NODATA value
    GEOMED_DATAIN nodata;
} geomed_option;

// read-only source image
typedef struct {
    // use "long" for linear index to prevent compiler related issues
    uint32_t width;
    uint32_t height;
    const GEOMED_DATAIN* raw;
} geomed_image;

void geomed_filter_new (geomed_filter *filter, uint32_t R);
void geomed_filter_del (geomed_filter *filter);
// the main routine
void DLL_DECLSPEC geomed (geomed_image image, geomed_mask mask, geomed_option option, GEOMED_DATAOUT* statsall);

#include "geomed3dv4_python.h"

#ifdef __cplusplus
}
#endif

#endif  // GEOMED_H

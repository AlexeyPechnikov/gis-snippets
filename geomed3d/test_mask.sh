#!/bin/sh
set -e

CC="gcc-10 -O2 -mmmx -msse -lm -funroll-all-loops -fopenmp -fPIC -D_FILE_OFFSET_BITS=64 -Wall"
CCC="$CC -o dynamically_linked -L. -lgeomed3dv4"

rm -f test_mask
$CCC test_mask.c -o test_mask
file test_mask

./test_mask 1

#!/bin/sh
set -e

CC="gcc-10 -O2 -mmmx -msse -lm -funroll-all-loops -fopenmp -fPIC -D_FILE_OFFSET_BITS=64 -Wall"
CCC="$CC -o dynamically_linked -L. -lgeomed3dv4 -lnetcdf"

rm -f libgeomed3dv4.so
$CC -shared geomed3dv4.c -o libgeomed3dv4.so

echo SUCCESS

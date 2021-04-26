#!/bin/sh
set -e

CC="gcc-10 -O2 -mmmx -msse -lm -funroll-all-loops -fopenmp -fPIC -D_FILE_OFFSET_BITS=64 -Wall"
CCC="$CC -o dynamically_linked -L. -lgeomed3dv4"

rm -f focal
$CCC focal.c -o focal
file focal

./focal 1

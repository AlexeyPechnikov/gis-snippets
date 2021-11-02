## Focal statistics processing library geomed3dv4.py

<img src="focal.gif" width=400>

The python script collects some helpful functions for VTK files export by numpy only and geo data processing and analysis. Also, it's wrapper for binary C coded library **libgeomed3dv4.so** which can be compiled from the sources by script [geomed3dv4.sh](geomed3dv4.sh) **focal** binary is test tool for the binary library to print the focal mask generating by the library. Use [focal.sh](focal.sh) to compile and [focal.ipynb](focal.ipynb) to visualize the output.

We have been used the library for many years to generate some complex statistics on big rasters (which are cropped from the shared code). For now, we've rewritten the code in Python using Numba + Dask for easy usage in Amazon and Google clouds, giving identical performance to the native code. The extension could be a great example of Python + C libraries interaction. 

## Compile and use geomed3 library on MacOS using HomeBrew GCC

See notebook [geomed3dv4.ipynb](geomed3dv4.ipynb) and the live notebook on Google Colab [Geophysical Inversion Library: GeoMed3d](https://colab.research.google.com/drive/1sle-WBlV_Z8bBv9dYpxe82FSevkJ6rWn?usp=sharing)

## Old focal statistics processing library geomed3dv3.py

Old library version 3 [geomed3dv3.py](geomed3dv3.py) provides some helpful functions to open raster and vector files by GDAL, produce ASCII or GeoTIFF files from Xarray DataSets, save datasets to PostgreSQL/PostGIS and so on. For now, that's easier to use Xarray+RasterIO and GeoPandas to do.

## Straight Line Hough Transform for Lineaments Caclulation

See the notebook
[Straight Line Hough Transform](Straight%20Line%20Hough%20Transform.ipynb)
![](Straight%20Line%20Hough%20Transform.jpg)

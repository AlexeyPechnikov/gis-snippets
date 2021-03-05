# GIS Snippets

This repository provides Python 3 Jupyter notebooks and code snippets for GIS tasks like to

# Google Earth Engine (GEE) Javascript code snippets

GEE is great and free of charge spatial processin engine which allows to process millions of available satellite images (Landsat 7/8, Sentinel 2 and much more) in minutes. We use GEE to produce high-quality satellite images and amplitude radar images (SAR) composites
and extract DEM and so on. See also our project repositories for GEE scripts.

<img src="GEE/Switzerland Mosaic using Google Earth Engine.jpg" width=400>

## Spatial data processing in PostgreSQL/PostGIS

Usually we use raster data processing whch is fast and well-optimized due to defined data grid.
Nevertheless, there are lots of cases when we require for sparse/irregular/multicale data processing.
It can be done dy different ways and PostgreSQL/PostGIS engine for data processing is powerful and helpful.

 * [Spheroid](Spheroid) based spatial data processing for irregular data.
 Here Radon Transform creates bins by radial distance from point and calculate average values for the bins and Gaussian Transform
 calculates transformation weights by distance on sphere (or ellipsoid). That's slow relative to regular grid processing but
 allow high accuracy processing on sparse data.

<img src="Spheroid/Gaussian%20Filtering%20on%20Spheroid%20%5BGlobal%5D.jpg" width=400>

<img src="Spheroid/Radon%20Transformation%20on%20Spheroid%20%5BGlobal%5D.jpg" width=400>

## Spatial spectrum processing of gravity, magnetic, DEM and satellite images

 * See [Super Resolution](Super Resolution) for gravity enhanced by DEM and [Elevation1 - Mashhad, Iran](Elevation1 - Mashhad, Iran) for DEM enhanced by satellite/orthophoto image by spatial spectrum transfer technique. Check the spatial correlation (coherence) and transfer the waveband with spectrum components amplitudes normalization.
<img src="Elevation1%20-%20Mashhad%2C%20Iran/Super-resolution%20DEM.3d.jpg" width=400>

 * [3D Seismic Spectral Components Analysis](3D Seismic Spectral Components Analysis) illustrates 3D spatial spectral analysis to check data quality and estimate the real spatial resolution of 3D seismic data
<img src="3D%20Seismic%20Spectral%20Components%20Analysis/3D%20Seismic%20Data.jpg" width=400>

## Amazon AWS EC2 initialization shell script for Ubuntu 18.04 LTS Bionic GIS installation

For our science and industrial data processing we often need thousands of processing cores and Amazon cloud engine (AWS) is 
the great choice for us. This [AWS init script](aws) allows to create simultaniously cluster of Amazon EC2 instances
with pre-configured Dask processing cluster and installled as geospatial and science processing libraries as PostgreSQL/PostGIS database engine.

 * ParaView Programmable Sources and Filters,
 * Fractality Index calculation by Gravity and DEM for density estimation,
 * Some well-known papers reproduced by spectral approach,
 * ,
 * etc.

# GIS Snippets

This repository provides Python 3 Jupyter notebooks and code snippets for GIS tasks like to

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


## Spatial spectrum processing of gravity, magnetic, DEM and satellite images,

 * See [Super Resolution](Super Resolution) for gravity enhanced by DEM and [Elevation1 - Mashhad, Iran](Elevation1 - Mashhad, Iran) for DEM enhanced by satellite/orthophoto image by spatial spectrum transfer technique. Check the spatial correlation (coherence) and transfer the waveband with spectrum components amplitudes normalization.
<img src="Elevation1%20-%20Mashhad%2C%20Iran/Super-resolution%20DEM.3d.jpg" width=400>

 * [3D Seismic Spectral Components Analysis](3D Seismic Spectral Components Analysis) illustrates 3D spatial spectral analysis to check data quality and estimate the real spatial resolution of 3D seismic data
<img src="3D%20Seismic%20Spectral%20Components%20Analysis/3D%20Seismic%20Data.jpg" width=400>

 * ParaView Programmable Sources and Filters,
 * Fractality Index calculation by Gravity and DEM for density estimation,
 * Some well-known papers reproduced by spectral approach,
 * Google Earth Engine Javascript snippets,
 * Amazon AWS EC2 initialization shell script for Ubuntu 18.04 LTS Bionic GIS installation,
 * etc.

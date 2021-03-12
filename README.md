# GIS Snippets

This repository provides Python 3 Jupyter notebooks and code snippets for our common tasks.

## Spectral approach on example of Antarctica region geological analysis

Here we extend analysis and visualization methods from "9.1 Computing Coherence on 2-D Grids" from "The IHO-IOC GEBCO Cook Book", contributed by K. M. Marks, NOAA Laboratory for Satellite Altimetry, USA. See also more detailed explanation of the original approach in a paper "Radially symmetric coherence between satellite gravity and multibeam bathymetry grids" (Marks and Smith, 2012).

For details see LinkedIn pulications [Spectral Coherence between Gravity and Bathymetry Grids](https://www.linkedin.com/pulse/computing-coherence-between-two-dimensional-gravity-grids-pechnikov/) and [The Density-Depth Model by Spectral Fractal Dimension Index](https://www.linkedin.com/pulse/density-model-spectral-fractal-dimension-index-alexey-pechnikov/)

<img src="Antarctica/Pearson%20Correlation%20Coefficient:%20GEBCO_2019%20vs%20Sandwell%20and%20Smith%20Gravity%20Anomaly.jpg" width=400>

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

 * See [Super Resolution](Super%20Resolution) for gravity enhanced by DEM and [Elevation1 - Mashhad, Iran](Elevation1 - Mashhad, Iran) for DEM enhanced by satellite/orthophoto image by spatial spectrum transfer technique. Check the spatial correlation (coherence) and transfer the waveband with spectrum components amplitudes normalization.
<img src="Elevation1%20-%20Mashhad%2C%20Iran/Super-resolution%20DEM.3d.jpg" width=400>

 * [3D Seismic Spectral Components Analysis](3D%20Seismic%20Spectral%20Components%20Analysis) illustrates 3D spatial spectral analysis to check data quality and estimate the real spatial resolution of 3D seismic data
<img src="3D%20Seismic%20Spectral%20Components%20Analysis/3D%20Seismic%20Data.jpg" width=400>

 * Mathematics of Gravity Inversion

See mathematical basics for gravity decomposition transforms applicable for Computer Vision (CV) methods.

[](Inversion%20of%20Gravity%20Mathematics)

<img src="Inversion%20of%20Gravity%20Mathematics/Spheres.jpg" width=400>

 * [3D Synthetic Density Inversion by Circular Hough Transform [Focal Average]](3D%20Synthetic%20Density%20Inversion%20by%20Circular%20Hough%20Transform%20%5BFocal%20Average%5D)

A set of 3D models for different synthetic density distributions inversion. In addition, the models include Density Inversion by Fractality Index.

<img src="3D%20Synthetic%20Density%20Inversion%20by%20Circular%20Hough%20Transform%20%5BFocal%20Average%5D/basic1to4.jpg" width=400>

## ParaView Programmable Sources and Filters

[ParaView](https://www.paraview.org) is the best 3D/4D data processing and visualization software which we know. And it's completely free  and Open Source and embeddes Python 3 interpretator and libraries for for advanced users. See our [ParaView snippets](ParaView) to do lot's of spatial data processing tasks like to loading different raster and vector data formats, reprojecting and so on inside ParaView. We use the snippets as is and modify them for some one-time tasks. Also, we provide separate repository [N-Cube ParaView plugin for 3D/4D GIS Data Visualization](https://github.com/mobigroup/ParaView-plugins) for common repeatable tasks like to DEM and satellite images and shapefiles 3D visualization, well logs visualization, table data mapping ans more.

<img src="https://github.com/mobigroup/gis-snippets/blob/master/ParaView/ProgrammableFilter/ParaView_ProgrammableFilter_reproject.jpg" width=400>

## Google Earth Engine (GEE) Javascript code snippets

GEE is great and free of charge spatial processin engine which allows to process millions of available satellite images (Landsat 7/8, Sentinel 2 and much more) in minutes. We use GEE to produce high-quality satellite images and amplitude radar images (SAR) composites
and extract DEM and so on. See also our project repositories for GEE scripts.

<img src="GEE/Switzerland Mosaic using Google Earth Engine.jpg" width=400>

## Amazon AWS EC2 initialization shell script for Ubuntu 18.04 LTS Bionic GIS installation

For our science and industrial data processing we often need thousands of processing cores and Amazon cloud engine (AWS) is 
the great choice for us. This [AWS init script](aws) allows to create simultaniously cluster of Amazon EC2 instances
with pre-configured Dask processing cluster and installled as geospatial and science processing libraries as PostgreSQL/PostGIS database engine.

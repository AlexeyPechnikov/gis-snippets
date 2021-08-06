#import os, sys, ctypes
#from numpy.ctypeslib import ndpointer
#import numpy as np
#import xarray as xr

import pandas as pd
# define Pandas display settings
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# it's similar to xr.open_rasterio() but attributes are different
# function to load source GeoTIF image
#def gdal_raster(src_filename, NoData=None)

# cell center (0.5, 0.5) should be pixel (0,0) but not rounded (1,1)
def geomed_round(arr):
    import numpy as np
    #return np.array([ (round(x,0)-1 if (x % 1 == 0.5) else round(x,0) ) for x in arr ]).astype(int)
    return np.array(arr).astype(int)

# main geomed library function for statistics calculations
def geomed(lib, raster, grid, radius_min, radius_max, gridded=False, scale_factor=0.707):
    import os, sys, ctypes
    from numpy.ctypeslib import ndpointer
    import numpy as np
    import xarray as xr
    import pandas as pd

    # build mask of input points
    _grid = grid.copy()
    # use zero surface if z is not defined
    if not 'z' in _grid:
        _grid['z'] = 0

    # prepare attributes
    if 'nodata' not in raster:
        raster.attrs['nodata'] = np.nan
    # see also raster.attrs['res']
    if 'transform' in raster.attrs:
        raster.attrs['ulx'] = raster.attrs['transform'][2]
        #float(raster.x.min()) - raster.attrs['transform'][0]/2
        raster.attrs['xres'] = raster.attrs['transform'][0]
        raster.attrs['lrx'] = raster.attrs['transform'][2]+raster.attrs['transform'][0]*raster.x.size
        #float(raster.x.max()) + raster.attrs['transform'][0]/2
        raster.attrs['yres'] = raster.attrs['transform'][4]
        raster.attrs['uly'] = raster.attrs['transform'][5]
        #float(raster.y.max()) - raster.attrs['transform'][4]/2
        raster.attrs['lry'] = raster.attrs['transform'][5]+raster.attrs['transform'][4]*raster.y.size
        #float(raster.y.min()) + raster.attrs['transform'][4]/2

    if gridded:
        mask = xr.Dataset.from_dataframe(_grid.set_index(['y','x']))
        mask['pixelx'] = geomed_round((mask.x - raster.ulx)/raster.xres)
        mask['pixely'] = geomed_round((mask.y - raster.uly)/raster.yres)
        # use zero surface depth instead of missed values
        mask.z.values = mask.z.fillna(0)
    else:
        _grid['pixelx'] = geomed_round((_grid.x - raster.ulx)/raster.xres)
        _grid['pixely'] = geomed_round((_grid.y - raster.uly)/raster.yres)
        mask = xr.Dataset.from_dataframe(_grid)
    del _grid

    if abs(np.round(raster.xres)) != abs(np.round(raster.yres)):
        raise Exception('The raster pixel x and pixel y resolutions must be ± equal')

    # define function to get stats count & names
    pygeomed_stats = lib.pygeomed_stats
    pygeomed_stat = lib.pygeomed_stat
    pygeomed_stat.restype = ctypes.c_char_p

    # define function to calculate focal statistics
    pygeomed = lib.pygeomed
    pygeomed.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                        ctypes.c_uint32,
                        ctypes.c_uint32,
                        ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_int32,flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_int32,flags="C_CONTIGUOUS"),
                        ctypes.c_uint32,
                        ctypes.c_uint32,
                        ctypes.c_uint32,
                        ctypes.c_float,
                        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
    pygeomed.restype = None

    # prepare input points mask for calculation function
    if gridded:
        mask_length = len(mask.pixelx)*len(mask.pixely)
        x, y = np.meshgrid(mask.pixelx, mask.pixely)
        x = (x).reshape((-1,mask_length))
        y = (y).reshape((-1,mask_length))
    else:
        mask_length = len(mask.pixelx)
        x = mask.pixelx.values
        y = mask.pixely.values
    z = mask.z.values.reshape((-1,mask_length))
    z = np.round(z/(abs(raster.xres)*0.7)).astype(int)
    zmax = int(np.max(z))

    # create output buffer for calculation function
    dataout = np.empty((mask_length,pygeomed_stats(),(radius_max-radius_min+1)),dtype=np.float32)

    # prepare source raster for calculation function
    datain = raster.values.astype(ctypes.c_float)

    # call calculation function
    pygeomed(datain, ctypes.c_uint32(raster.shape[1]), ctypes.c_uint32(raster.shape[0]),
            x.astype(ctypes.c_int32),y.astype(ctypes.c_int32),z.astype(ctypes.c_int32),ctypes.c_uint32(mask_length),
            ctypes.c_uint32(radius_min),ctypes.c_uint32(radius_max),ctypes.c_float(raster.nodata),
            dataout)
    # prepared buffer for source raster is not required later
    del datain

    # define data variables for NetCDF dataset
    statnames = []
    datavars = {}
    if gridded:
        dataout = dataout.reshape((pygeomed_stats(),(radius_max-radius_min+1),len(mask.y),len(mask.x)))
        dims = ['z','y','x']
    else:
        dataout = dataout.reshape((pygeomed_stats(),(radius_max-radius_min+1),mask_length))
        dims = ['z','l']
        datavars['y'] = (['l'],mask.y)
        datavars['x'] = (['l'],mask.x)
        datavars['surface'] = (['l'],mask.z)
    for statidx in range(0,pygeomed_stats()):
        if sys.version_info >= (3, 0):
            statname = "".join(map(chr, pygeomed_stat(statidx)))
        else:
            statname = pygeomed_stat(statidx)
        datavars[statname] = (dims,dataout[statidx,:,:])
    del dataout

    # build NetCDF dataset
    if gridded:
        ds = xr.Dataset(datavars,
            coords={
                'surface': mask.z,
                'z': np.arange(radius_min,radius_max+1)[::-1]
            }
        )
    else:
        ds = xr.Dataset(datavars,
            coords={
                'l': 1.*np.arange(0,mask_length),
                'z': np.arange(radius_min,radius_max+1)[::-1]
            }
        )
        # change lat/lon variables to coordinates
        ds.coords['y'] = ds.data_vars['y']
        ds.coords['x'] = ds.data_vars['x']
        ds.coords['surface']= ds.data_vars['surface']
        # length per profile
        ds.l.values[1:] = np.cumsum(np.sqrt(np.diff(ds.y.values)**2 + np.diff(ds.x.values)**2))
    del datavars

    # set real depth (negative)
    ds['z'] = (scale_factor*abs(raster.xres))*(zmax-ds.z.values)

    # add projection information from source raster to NetCDF dataset
    epsg=np.int32(raster.epsg if 'epsg' in raster and raster.epsg is not None and raster.epsg != '' else 0)

    ds.attrs['epsg'] = epsg
    ds['projection']=''
    if 'spatial_ref' in raster.attrs:
        ds.projection.attrs['spatial_ref'] = raster.attrs['spatial_ref']
    ds.coords['projection'] = ds.data_vars['projection']
    for datavar in ds.data_vars:
        ds[datavar].attrs = {'grid_mapping': 'projection', 'epsg': epsg}

    # return NetCDF dataset
    return ds

#https://en.wikipedia.org/wiki/Gaussian_filter
#https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
#from scipy.ndimage.filters import gaussian_filter

def raster_gamma_range(raster0, g1, g2, backward=False):
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter

    raster = raster0.copy()
    raster.values = raster.values.astype(np.float32)
    if backward:
        raster.values = gaussian_filter(raster.values,g1) \
            - gaussian_filter(raster.values,g2)
    else:
        raster.values = gaussian_filter(raster.values,g1,mode='constant', cval=np.nan) \
            - gaussian_filter(raster.values,g2,mode='constant', cval=np.nan)
    return raster

def raster_gamma(raster0, g, backward=False):
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter

    raster = raster0.copy()
    if backward:
        raster.values = gaussian_filter(raster.values.astype(np.float32),g)
    else:
        raster.values = gaussian_filter(raster.values.astype(np.float32),g,mode='constant', cval=np.nan)
    return raster


# deprecated - use dem.interp_like(raster, 'linear') or dem.interp_like(raster, 'nearest') instead
#def dem2df(dem, _df):

def vtk2da(filename, varname='None'):
    from vtk import vtkStructuredPointsReader
    from vtk.util import numpy_support as VN
    import numpy as np
    import numpy as np
    import xarray as xr

    reader = vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()
    dim = data.GetDimensions()
    bnd = data.GetBounds()
    values = VN.vtk_to_numpy(data.GetPointData().GetArray(varname))
    values = values.reshape(dim,order='F')

    da = xr.DataArray(values.transpose([2,1,0]),
                coords=[np.linspace(bnd[4],bnd[5],dim[2]),
                        np.linspace(bnd[2],bnd[3],dim[1]),
                        np.linspace(bnd[0],bnd[1],dim[0])],
                dims=['z','y','x'])
    return da


### Save to VTK (version 1) files
def da2vtk_scalar(da, filename, filter_by_output_range=None):
    import numpy as np
    import sys

    vals = da.values
    vals = 100.*(vals - np.nanmin(vals))/(np.nanmax(vals)-np.nanmin(vals))
    if not filter_by_output_range is None:
        vals[(vals<filter_by_output_range[0])|(vals>filter_by_output_range[1])] = np.nan
        vals = 100.*(vals - np.nanmin(vals))/(np.nanmax(vals)-np.nanmin(vals))
    # Use "A*(A/A)" expression in Voxler 4 "math" unit
    #vals[np.isnan(vals)] = 0
    #vals[vals==0] = np.nan

    header = """# vtk DataFile Version 1.0
vtk output
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS %d %d %d
ASPECT_RATIO %f %f %f
ORIGIN %f %f %f
POINT_DATA %d
SCALARS %s float
LOOKUP_TABLE default
""" % (da.x.shape[0],da.y.shape[0],da.z.shape[0],
                (np.nanmax(da.x.values)-np.nanmin(da.x.values))/(da.x.shape[0]-1),
                (np.nanmax(da.y.values)-np.nanmin(da.y.values))/(da.y.shape[0]-1),
                (np.nanmax(da.z.values)-np.nanmin(da.z.values))/(da.z.shape[0]-1),
                np.nanmin(da.x.values),
                np.nanmin(da.y.values),
                np.nanmin(da.z.values),
                da.x.shape[0]*da.y.shape[0]*da.z.shape[0],
                da.name)

    with open(filename, 'wb') as f:
        if sys.version_info >= (3, 0):
            f.write(bytes(header,'utf-8'))
        else:
            f.write(header)
        np.array(vals, dtype=np.float32).byteswap().tofile(f)

### Save vector with components (i,j,k) to VTK (version 4.2) binary files
# ds2vtk3(ds, 'velocity', fname + '.vtk')
def ds2vtk_vector(ds, name, filename):
    import numpy as np
    import sys

    da = ds.transpose('z','y','x')
    header = """# vtk DataFile Version 4.2
vtk output
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS %d %d %d
SPACING %f %f %f
ORIGIN %f %f %f
POINT_DATA %d
VECTORS %s float
""" % (da.x.shape[0],da.y.shape[0],da.z.shape[0],
                (np.nanmax(da.x.values)-np.nanmin(da.x.values))/(da.x.shape[0]-1),
                (np.nanmax(da.y.values)-np.nanmin(da.y.values))/(da.y.shape[0]-1),
                (np.nanmax(da.z.values)-np.nanmin(da.z.values))/(da.z.shape[0]-1),
                np.nanmin(da.x.values),
                np.nanmin(da.y.values),
                np.nanmin(da.z.values),
                da.x.shape[0]*da.y.shape[0]*da.z.shape[0],
                name)

    with open(filename, 'wb') as f:
        f.write(bytes(header,'utf-8'))
        arr = np.stack([da.i.values, da.j.values, da.k.values],axis=-1)
        np.array(arr, dtype=np.float32).byteswap().tofile(f)

def da2vtk1_scalar_int(da, filename):
    import numpy as np
    import sys

    vals = da.values
    header = """# vtk DataFile Version 1.0
vtk output
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS %d %d %d
ASPECT_RATIO %f %f %f
ORIGIN %f %f %f
POINT_DATA %d
SCALARS %s int32
LOOKUP_TABLE default
""" % (da.x.shape[0],da.y.shape[0],da.z.shape[0],
                (np.nanmax(da.x.values)-np.nanmin(da.x.values))/(da.x.shape[0]-1),
                (np.nanmax(da.y.values)-np.nanmin(da.y.values))/(da.y.shape[0]-1),
                (np.nanmax(da.z.values)-np.nanmin(da.z.values))/(da.z.shape[0]-1),
                np.nanmin(da.x.values),
                np.nanmin(da.y.values),
                np.nanmin(da.z.values),
                da.x.shape[0]*da.y.shape[0]*da.z.shape[0],
                da.name)

    with open(filename, 'wb') as f:
        if sys.version_info >= (3, 0):
            f.write(bytes(header,'utf-8'))
        else:
            f.write(header)
        np.array(vals, dtype=np.int32).byteswap().tofile(f)

def unit_circle_2d(r):
    import numpy as np

    A = np.arange(-r,r+1)**2
    dists = np.sqrt( A[:,None] + A)
    # circle
    #return (np.abs(dists-r)<=0).astype(int)
    # filled circle
    if r <= 2:
        return ((dists-r)<=0).astype(int)
    return ((dists-r)<=0.5).astype(int)

# z, y, x
#footprint = np.array((2*rz+1)*[unit_circle_2d(r)])
#print (footprint.shape)
#plt.imshow(footprint[0], interpolation='None')

def unit_ring_2d(r):
    import numpy as np

    A = np.arange(-r,r+1)**2
    dists = np.sqrt( A[:,None] + A)
    if r <= 2:
        return (np.abs(dists-r)<=0).astype(int)
    return (np.abs(dists-r)<=0.5).astype(int)

# y, x
#footprint = unit_ring_2d(4)
#print (footprint.shape)
#plt.imshow(footprint, interpolation='None')

# GEE helper functions
#import urllib
#import shutil
#import ee


# Function to mask clouds using the Sentinel-2 QA band.
def GEEmaskS2clouds(image):
    # Get the pixel QA band.
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    cloudMask = qa.bitwiseAnd(cloudBitMask).eq(0)
    cirrusMask = qa.bitwiseAnd(cirrusBitMask).eq(0)

    # Return the masked and scaled data, without the QA bands.
    return image\
        .updateMask(cloudMask)\
        .updateMask(cirrusMask)\
        .divide(10000)\
        .select("B.*")\
        .copyProperties(image, ["system:time_start"])

# Function to mask edges on Sentinel-1 GRD image
def GEEmaskS1edges(image):
    edge = image.lt(-30.0)
    maskedImage = image.mask().And(edge.Not())
    return image.updateMask(maskedImage)

# works for GEE geographic coordinates only
def gee_image2rect(GEEimage, reorder=False):
    import numpy as np

    coords = GEEimage.getInfo()['properties']['system:footprint']['coordinates'][0]
    lats = np.asarray(coords)[:,1]
    lons = np.asarray(coords)[:,0]
    if not reorder:
        return [lons.min(), lats.min(), lons.max(), lats.max()]
    else:
        return [lons.min(), lons.max(), lats.min(), lats.max()]

# create worldfile to define image coordinates
def worldfile_tofile(fname, GEEimage, dimensions):
    import os

    area = gee_image2rect(GEEimage)
    name, ext = os.path.splitext(fname)
    # use QGIS worldfile names convention 
    jext = ext[1] + ext[-1] + 'w'
    fname = os.path.join(str(os.extsep).join([name,jext]))
    with open(fname, 'w') as outfile:
        xres = (area[2]-area[0])/dimensions[0]
        yres = (area[1]-area[3])/dimensions[1]
        coefficients = [xres, 0, 0, yres, area[0], area[3]]
        print('\n'.join(map(str, coefficients)), file=outfile)

# download GEE URL and save to file
def geeurl_tofile(GEEurl, fname):
    import urllib
    import shutil

    with urllib.request.urlopen(GEEurl) as response, open(fname, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)

def gee_preview_tofile(GEEimage, vis, dimensions, fname=None):
    GEEurl = GEEimage\
        .visualize(**vis)\
        .getThumbURL({'dimensions':dimensions, 'format': 'jpg'})
    #print (GEEurl)
    if fname is not None:
        geeurl_tofile(GEEurl, fname)
        worldfile_tofile(fname, GEEimage, dimensions)
    return {'url': GEEurl, 'width': dimensions[0], 'height': dimensions[1]}

def split_rect(GEEimage, n):
    rect = gee_image2rect(GEEimage)
    lats = np.linspace(rect[1], rect[3], n+1)
    lons = np.linspace(rect[0], rect[2], n+1)
    #print (lats, lons)
    cells = []
    for lt1, lt2 in zip(lats.ravel()[:-1], lats.ravel()[1:]):
        for ll1, ll2 in zip(lons.ravel()[:-1], lons.ravel()[1:]):
            cell = [lt1, ll1, lt2, ll2]
            cells.append(cell)
    return cells

def zipsbands2image(files):
    import xarray as xr
    import zipfile

    dss = []
    # merge separate file areas
    for fname in sorted(files):
        #print ('fname', fname)
        zip = zipfile.ZipFile(fname)
        # merge separate file to dataset
        ds = xr.Dataset()
        for bandname in zip.namelist():
            varname = bandname.split('.')[1]
            da = xr.open_rasterio(f'/vsizip/{fname}/{bandname}').squeeze(drop=True)
            ds[varname] = da
            da.close()
        dss.append(ds)
    return xr.merge(dss)

#writer.WriteToOutputStringOn()
#writer.Write()
#binary_string = writer.GetBinaryOutputString()
def vtkpoints2ds(filename_or_binarystring):
    import xarray as xr
    import numpy as np
    #from vtk import vtkStructuredGridReader
    from vtk import vtkStructuredPointsReader
    from vtk.util import numpy_support as VN

    reader = vtkStructuredPointsReader()
    if type(filename_or_binarystring) == bytes:
        reader.ReadFromInputStringOn()
        reader.SetBinaryInputString(filename_or_binarystring, len(filename_or_binarystring))
    else:
        reader.SetFileName(filename_or_binarystring)
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()
    dim = data.GetDimensions()
    bnd = data.GetBounds()

    points = data.GetPointData()

    ds = xr.Dataset()
    for idx in range(points.GetNumberOfArrays()):
        arrayname = points.GetArrayName(idx)
        values = VN.vtk_to_numpy(points.GetArray(arrayname))
        values = values.reshape(dim,order='F')

        da = xr.DataArray(values.transpose([2,1,0]),
                    coords=[np.linspace(bnd[4],bnd[5],dim[2]),
                            np.linspace(bnd[2],bnd[3],dim[1]),
                            np.linspace(bnd[0],bnd[1],dim[0])],
                    dims=['z','y','x'])
        ds[arrayname] = da

    return ds

def rasterize(image, areas, with_nodata=False):
    import xarray as xr
    from rasterio import features
    # increment class value to use 0 as placeholder later
    if 'class' in areas:
        geoms = [(g,c+1) for g,c in zip(areas['geometry'], areas['class'])]
    else:
        geoms = [(g,1) for g in areas['geometry']]
    # rasterio transform is broken, we need to build it from image extent
    # note: gdal uses pixel borders and xarray uses pixel centers
    if 'latitude' in image:
        band = 'latitude'
    else:
        # suppose the same geometries per bands
        band = list(image.data_vars)[0]
    #res = image[band].attrs['res']
    # be careful with ordering
    res = [float(image[band].x.diff('x')[0]), float(image[band].y.diff('y')[0])]
    xmin = image[band].x.values.min()
    ymax = image[band].y.values.max()
    transform = [res[0], 0, xmin - res[0]/2, 0, -res[1], ymax+res[1]/2]
    # rasterize geometries
    da = xr.zeros_like(image[band]).rename('class').astype(np.uint8)
    da.values = np.flipud(features.rasterize(geoms,
                              dtype=np.uint8,
                              out_shape=image[band].shape,
                              transform=transform)) - 1
    df = da.to_dataframe().reset_index()
    if not with_nodata:
        # remove placeholder zero value
        df = df[df['class']<255]
    # return dataarray with placeholder 255 and dataframe
    return da, df

# vectorize geometries on dask dataarray
def vectorize(image):
    from rasterio import features
    import geopandas as gpd

    # be careful with ordering
    res = [float(image.x.diff('x').mean()), float(image.y.diff('y').mean())]
    xmin = image.x.values.min()
    ymax = image.y.values.min()
    transform = [res[0], 0, xmin - res[0]/2, 0, res[1], ymax+res[1]/2]
    transform

    geoms = (
            {'properties': {'class': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(features.shapes(image.values, mask=None, transform=transform))
    )
    gdf = gpd.GeoDataFrame.from_features(geoms)

    return gdf

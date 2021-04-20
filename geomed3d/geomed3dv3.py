"""
By default GDAL uses Pillow and Pillow uses it's own version of libtiff when GDAL uses the system one.
And as result we have segfaults on some TIFFs in jupyter notebooks. Maybe resolved by:

pip2 uninstall Pillow
pip2 install --no-binary :all: Pillow
pip3 uninstall Pillow
pip3 install --no-binary :all: Pillow

brew uninstall gdal
brew install gdal
#Homebrew: gdal 2.4.4_4 is already installed
pip3 install GDAL

pip3 uninstall psycopg2
pip3 install psycopg2

pip3.7 install vtk
pip3.7 install rasterio
"""
from osgeo import osr, gdal, ogr
import os, sys, ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import xarray as xr
import pandas as pd

# it's similar to xr.open_rasterio() but attributes are different
# function to load source GeoTIF image
def gdal_raster(src_filename, NoData=None):
    ds = gdal.Open(src_filename)
    datains = []
    NoDatas = []
    for bandidx in range(ds.RasterCount):
        # read NoData value from raster (if possible)
        band = ds.GetRasterBand(bandidx+1)
        datain = np.array(band.ReadAsArray())
        if NoData is None:
            nodatain = band.GetNoDataValue()
            if nodatain is not None and datain.dtype in ['float32','float64']:
                NoData = nodatain
            elif nodatain is not None:
                # gdal returns float NoData value for integer bands
                NoData = int(nodatain) if int(nodatain) == nodatain else nodatain
            else:
                NoData = 0
        datains.append(datain)
        NoDatas.append(NoData)
    if len(datains) == 1:
        NoDatas = NoDatas[0]
        raster = xr.DataArray(datains[0],
                          coords=[range(ds.RasterYSize),range(ds.RasterXSize)],
                          dims=['y','x'])
    else:
        if np.all(NoDatas) == NoDatas[0]:
            NoDatas = NoDatas[0]
        else:
            NoDatas = np.array(NoDatas)
        raster = xr.DataArray(datains,
                          coords=[range(ds.RasterCount),range(ds.RasterYSize),range(ds.RasterXSize)],
                          dims=['band','y','x'])

    wkt = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)

    if 'EPSG' == srs.GetAttrValue("AUTHORITY", 0):
        epsg = srs.GetAttrValue("AUTHORITY", 1)
    else:
        epsg = ''

    ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
    lrx = ulx + (ds.RasterXSize - 1) * xres
    lry = uly + (ds.RasterYSize - 1) * yres

    raster['y'] = uly + yres*(raster.y.values + 0.5)
    raster['x'] = ulx + xres*(raster.x.values + 0.5)
    raster.attrs['nodata'] = NoDatas
    raster.attrs['ulx'] = ulx
    raster.attrs['xres'] = xres
    raster.attrs['xskew'] = xskew
    raster.attrs['uly'] = uly
    raster.attrs['yskew'] = yskew
    raster.attrs['yres'] = yres
    raster.attrs['lrx'] = lrx
    raster.attrs['lry'] = lry
    raster.attrs['spatial_ref'] = wkt
    raster.attrs['epsg'] = epsg

    return raster

"""
raster = gdal_raster("IMAGE_HH_SRA_wide_001.tif")
raster

<xarray.DataArray (y: 17366, x: 20633)>
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint16)
Coordinates:
  * y      (y) float64 4.66e+05 4.66e+05 4.66e+05 4.66e+05 4.659e+05 ...
  * x      (x) float64 3.69e+05 3.69e+05 3.69e+05 3.69e+05 3.691e+05 ...
Attributes:
    nodata:   0
    ulx:      368992.5
    xres:     15.0
    xskew:    0.0
    uly:      466007.5
    yskew:    0.0
    yres:     -15.0
    lrx:      678487.5
    lry:      205517.5
"""

# function to load 2D/3D points, lines, polygons from shapefile
# see output "proj" attribute to check projection
def ogr_vector(shapefile):
    'Given a shapefile path, return a list of 3D points in GIS coordinates'
    shapeData = ogr.Open(shapefile)
    if not shapeData:
        raise Exception('The shapefile is invalid')
    # Make sure there is exactly one layer
    if shapeData.GetLayerCount() != 1:
        raise Exception('The shapefile must have exactly one layer')
    # Get the first layer
    layer = shapeData.GetLayer()
    # get all field names
    layerDefinition = layer.GetLayerDefn()
    fieldnames = []
    for i in range(layerDefinition.GetFieldCount()):
        fieldname = layerDefinition.GetFieldDefn(i).GetName()
        fieldnames.append(fieldname)
    # process all features in the layer
    points = []
    # For each point,
    for index in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(index)
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        gtype = geometry.GetGeometryType()

        fields = {}
        for fieldname in fieldnames:
            fields[fieldname] = feature.GetField(fieldname)
            #print fieldname, feature.GetField(fieldname)

        if gtype in [ogr.wkbPoint25D, ogr.wkbPoint]:
            pointCoordinates = dict(x=geometry.GetX(), y=geometry.GetY(), z=geometry.GetZ())
            points.append(dict(pointCoordinates,**fields))
        elif gtype in [ogr.wkbLineString, ogr.wkbLineString25D]:
            for point in range(geometry.GetPointCount()):
                pointCoordinates = dict(x=geometry.GetX(point), y=geometry.GetY(point), z=geometry.GetZ(point))
                points.append(dict(pointCoordinates,**fields))
        elif gtype in [ogr.wkbPolygon, ogr.wkbPolygon25D]:
            # extract boundary box
            (minX, maxX, minY, maxY, minZ, maxZ) = geometry.GetEnvelope3D()
            pointCoordinates = dict(x=minX, y=minY, z=minZ)
            points.append(dict(pointCoordinates,**fields))
            pointCoordinates = dict(x=maxX, y=maxY, z=maxZ)
            points.append(dict(pointCoordinates,**fields))
        else:
            raise Exception('This module can only load points, lines and polygons')

        feature.Destroy()
    # Get spatial reference as proj4
    if layer.GetSpatialRef() is None:
        proj4 = ''
    else:
        proj4 = layer.GetSpatialRef().ExportToProj4()
    shapeData.Destroy()
    #points = np.array(points)
    #df = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2]})
    df = pd.DataFrame(points)
    # add "proj" attribute to output dataframe
    df.proj4 = proj4
    return df


"""
df = ogr_vector("test_points/test_points.shp")
df.head()

Id	gsAttrib	x	y	z
0	0	0.040432	469827.964459	390884.634456	0.040432
1	1	0.434915	470083.763310	390884.634456	0.434915
2	2	0.758500	470339.562162	390884.634456	0.758500
3	3	0.488747	470595.361013	390884.634456	0.488747
4	4	0.945799	470851.159865	390884.634456	0.945799
"""

# cell center (0.5, 0.5) should be pixel (0,0) but not rounded (1,1)
def geomed_round(arr):
    #return np.array([ (round(x,0)-1 if (x % 1 == 0.5) else round(x,0) ) for x in arr ]).astype(int)
    return np.array(arr).astype(int)

# main geomed library function for statistics calculations
def geomed(lib, raster, grid, radius_min, radius_max, gridded=False, scale_factor=0.707):
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

    # prepate source raster for calculation function
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



# libraries to work with PostgreSQL database
import psycopg2
# https://stackoverflow.com/questions/11914472/stringio-in-python3
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def ds2postgis(conn, ds, table):
    # PostgreSQL cursor to execute SQL commands
    cur = conn.cursor()
    cur.execute("""
    DROP TABLE IF EXISTS """ + table + """;
    CREATE TABLE """ + table + """ (
        z double precision,
        y double precision,
        x double precision,
        """ + ','.join([x + ' double precision' for x in ds.data_vars]) + """
    )
    WITH OIDS;
    """)

    # process dataset
    data = np.array([ds[s].values for s in ds.data_vars])
    zvals = ds.z.values
    yvals = ds.y.values
    xvals = ds.x.values

    def _ds2postgis(datastring):
        # Convert string to file
        pg_input = StringIO( "\n".join(datastring) )
        pg_input.seek(0)
        # Load CSV data to PostgreSQL
        cur.copy_expert("COPY " + table + " FROM STDIN DELIMITER AS ',' NULL 'nan'", pg_input)

    if 'l' in ds.coords:
        # 1D & 2D
        lvals = ds.l.values
        # build CSV datastring
        datastring = []
        for lidx,l in enumerate(lvals):
            for zidx, z in enumerate(zvals):
                line = ','.join([str(v) for v in (z,yvals[lidx],xvals[lidx])] + [str(v) for v in data[:,zidx,lidx]])
                datastring.append( line )
        _ds2postgis(datastring)
    else:
        # 3D
        for zidx, z in enumerate(zvals):
            # build CSV datastring
            datastring = []
            for yidx,y in enumerate(yvals):
                for xidx,x in enumerate(xvals):
                    line = ','.join([str(v) for v in (z,y,x)] + [str(v) for v in data[:,zidx,yidx,xidx]])
                    datastring.append( line )
            _ds2postgis(datastring)

    # Add spatial column to the table
    cur.execute("ALTER TABLE " + table + " ADD COLUMN geom GEOMETRY;")
    cur.execute("UPDATE " + table + " SET geom=ST_SetSRID(ST_Point(x,y)," + str(ds.epsg.values) + ");")

    conn.commit()
    cur.close()

"""
# Save as PostGIS wide tables

import psycopg2
# String to connect to PostgreSQL database
connstring = "dbname='mbg' user='mbg' host='localhost' password=''"
# Connect to PostgreSQL
conn = psycopg2.connect(connstring)

ds2postgis(conn, ds1d, 'ds2d')
# Retrieve saved data from PostgreSQL
df1d = pd.read_sql_query("SELECT oid, * FROM ds1d ORDER by oid LIMIT 10", conn, coerce_float=True)
df1d.head()

ds2postgis(conn, ds2d, 'ds1d')
# Retrieve saved data from PostgreSQL
df2d = pd.read_sql_query("SELECT oid, * FROM ds2d ORDER BY oid LIMIT 10", conn, coerce_float=True)
df2d.head()

ds2postgis(conn, ds3d, 'ds3d')
# Retrieve saved data from PostgreSQL
df3d = pd.read_sql_query("SELECT oid, * FROM ds3d ORDER BY oid LIMIT 10", conn, coerce_float=True)
df3d.head()
"""

#q = [25,75]
def ds_percentile(ds, q):
    ds_q23 = ds.copy(deep=True)
    for stat in ds.data_vars:
        if stat == 'orig':
            continue
        pcnt = np.nanpercentile(ds_q23[stat].values.reshape(-1),q)
        ds_q23[stat].values = np.clip(ds_q23[stat].values,pcnt[0],pcnt[1])
    return ds_q23

# Z-Minus
#q = [25,75]
def ds_minus(ds, q=None):
    ds_minus = ds.copy(deep=True)
    for stat in ds.data_vars:
        if stat == 'orig':
            continue

        # depth and r orders are reverted so X(R)-X(R-1) <=> X(z-1)-X(z)
        arr0 = ds_minus[stat]
        arr = np.nan*np.zeros(arr0.shape)
        for z in range(1,arr0.shape[0]):
            arr[z,:] = arr0[z-1,:] - arr0[z,:]
        ds_minus[stat].values = arr

        if q is not None:
            pcnt = np.nanpercentile(ds_minus[stat].values.reshape(-1),q)
            ds_minus[stat].values = np.clip(ds_minus[stat].values,pcnt[0],pcnt[1])
    return ds_minus

# Z-Minus
#q = [25,75]
def da_minus(da, q=None):
    da_minus = da.copy(deep=True)

    # depth and r orders are reverted so X(R)-X(R-1) <=> X(z-1)-X(z)
    arr = np.nan*np.zeros(da_minus.shape)
    for z in range(1,da_minus.values.shape[0]):
        arr[z,:] = da_minus.values[z-1,:] - da_minus.values[z,:]

    if q is not None:
        pcnt = np.nanpercentile(arr.reshape(-1),q)
        arr = np.clip(arr,pcnt[0],pcnt[1])

    da_minus.values = arr
    return da_minus

# Z-Plus
#q = [25,75]
def ds_plus(ds, q=None):
    ds_plus = ds.copy(deep=True)
    for stat in ds.data_vars:
        if stat == 'orig':
            continue

        arr0 = ds_plus[stat]
        arr = np.nan*np.zeros(arr0.shape)
        for z in range(1,arr0.shape[0]):
            arr[z,:] = (arr0[z-1,:] + arr0[z,:])/2.
        ds_plus[stat].values = arr

        if q is not None:
            pcnt = np.nanpercentile(ds_plus[stat].values.reshape(-1),q)
            ds_plus[stat].values = np.clip(ds_plus[stat].values,pcnt[0],pcnt[1])
    return ds_plus

#https://en.wikipedia.org/wiki/Gaussian_filter
#https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
from scipy.ndimage.filters import gaussian_filter

# raster = raster_gamma_range(raster0, 11, 20)
#def raster_gamma_range(raster0, g1, g2, compress=False):
#    raster = raster0.copy()
#    raster.values = raster.values.astype(np.float32)
#    raster.values = gaussian_filter(raster.values,g1) - gaussian_filter(raster.values,g2)
#    if compress:
#        raise ValueError('"compress" option is disabled')
#        #raster.values = np.sign(raster.values)*np.sqrt(np.abs(raster.values))
#    return raster
def raster_gamma_range(raster0, g1, g2, backward=False):
    raster = raster0.copy()
    raster.values = raster.values.astype(np.float32)
    if backward:
        raster.values = gaussian_filter(raster.values,g1) \
            - gaussian_filter(raster.values,g2)
    else:
        raster.values = gaussian_filter(raster.values,g1,mode='constant', cval=np.nan) \
            - gaussian_filter(raster.values,g2,mode='constant', cval=np.nan)
    return raster

# raster = raster_gamma(raster0, 11)
#def raster_gamma(raster0, g, compress=False):
#    raster = raster0.copy()
#    raster.values = gaussian_filter(raster.values.astype(np.float32),g)
#    if compress:
#        raise ValueError('"compress" option is disabled')
#        #raster.values = np.sign(raster.values)*np.sqrt(np.abs(raster.values))
#    return raster
def raster_gamma(raster0, g, backward=False):
    raster = raster0.copy()
    if backward:
        raster.values = gaussian_filter(raster.values.astype(np.float32),g)
    else:
        raster.values = gaussian_filter(raster.values.astype(np.float32),g,mode='constant', cval=np.nan)
    return raster


#https://en.wikipedia.org/wiki/Web_Mercator#EPSG:3785
#http://gis.stackexchange.com/questions/62343/how-can-i-convert-a-ascii-file-to-geotiff-using-python
def ds2gtif_south(data, filename):
    coordz = list(data.coords)[0]
    coordl = list(data.coords)[1]

    shape = data.shape
    pixelz = round(data[coordz].values[1]-data[coordz].values[0],5)
    pixell = round(data[coordl].values[1]-data[coordl].values[0],5)

    types = ['uint8','uint16','int16','uint32','int32','float32','float64']
    gtypes = [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
             gdal.GDT_Float32, gdal.GDT_Float64]

    dtype = data.values.dtype
    tidx = types.index(dtype)
    gtype = gtypes[tidx]
    if tidx in [0,1,2,3,4]:
        nodata = np.iinfo(dtype).min
    else:
        nodata = 170141000918780003225695629360656023552.000

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(filename, shape[1], shape[0], 1, gtype, options = [ 'COMPRESS=LZW' ])
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    if data[coordz].values[0] < data[coordz].values[-1]:
        zlim = min(data[coordz].values)-pixelz/2
    else:
        zlim = max(data[coordz].values)-pixelz/2
    dst.SetGeoTransform( [ min(data[coordl].values)-pixell/2, pixell, 0, zlim, 0, pixelz ] )
    if 'epsg' in data and data.epsg is not None:
        srs = osr.SpatialReference()
        #srs.SetWellKnownGeogCS("WGS84")
        srs.ImportFromEPSG(int(data.epsg))
        dst.SetProjection( srs.ExportToWkt() )

    arr = data.values.copy()
    arr[np.isnan(arr)] = nodata

    dst.GetRasterBand(1).SetNoDataValue(nodata)
    dst.GetRasterBand(1).WriteArray(arr)

# north semisphere, usually increasing x,y order
def ds2gtif_north(data, filename):
    coordz = list(data.coords)[0]
    coordl = list(data.coords)[1]

    shape = data.shape
    pixelz = round(data[coordz].values[1]-data[coordz].values[0],5)
    pixell = round(data[coordl].values[1]-data[coordl].values[0],5)

    types = ['uint8','uint16','int16','uint32','int32','float32','float64']
    gtypes = [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
             gdal.GDT_Float32, gdal.GDT_Float64]

    dtype = data.values.dtype
    tidx = types.index(dtype)
    gtype = gtypes[tidx]
    if tidx in [0,1,2,3,4]:
        nodata = np.iinfo(dtype).min
    else:
        nodata = 170141000918780003225695629360656023552.000

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(filename, shape[1], shape[0], 1, gtype, options = [ 'COMPRESS=LZW' ])
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    if data[coordz].values[0] < data[coordz].values[-1]:
        zlim = max(data[coordz].values)+pixelz/2
    else:
        zlim = min(data[coordz].values)+pixelz/2

    dst.SetGeoTransform( [ min(data[coordl].values)-pixell/2, pixell, 0, zlim, 0, -pixelz ] )
    if 'epsg' in data and data.epsg is not None:
        srs = osr.SpatialReference()
        #srs.SetWellKnownGeogCS("WGS84")
        srs.ImportFromEPSG(int(data.epsg))
        dst.SetProjection( srs.ExportToWkt() )

    arr = np.flipud(data.values.copy())
    arr[np.isnan(arr)] = nodata

    dst.GetRasterBand(1).SetNoDataValue(nodata)
    dst.GetRasterBand(1).WriteArray(arr)

#ds2gtif_north(ds3d.orig[10], 'TX_AllenDome/test.tif')

def ds2ascii(ds, stat, depth, filename):
    #nodata = 1.70141000918780003225695629360656023552e38
    nodata = 170141000918780003225695629360656023552.000

    # ignore depth when there is no 'z' dimention
    if 'z' in list(ds.dims):
        plan = ds[stat].sel(z=depth,method='nearest')
    else:
        plan = ds[stat]

    minx = np.min(plan.x.values)
    miny = np.min(plan.y.values)

    pixelx = np.diff(plan.x.values)[0]
    pixely = np.diff(plan.y.values)[0]
    assert( abs(pixelx) == abs(pixely) )

    if pixely < 0:
        values = np.flipud(plan.values)
    else:
        values = plan.values

    height = plan.shape[0]
    width  = plan.shape[1]

    f = open(filename, 'w')

    f.write("ncols         %i\r\n" % width);
    f.write("nrows         %i\r\n" % height);
    f.write("xllcorner     %f\r\n" % (minx-pixelx/2));
    # TODO: CHECK FOR pixely > 0
    if pixely < 0:
        f.write("yllcorner     %f\r\n" % (miny+pixely/2));
    else:
        f.write("yllcorner     %f\r\n" % (miny-pixely/2));
    f.write("cellsize      %f\r\n" % pixelx);
    f.write("NODATA_value  %f\r\n" % nodata);
    for h in range(0,height):
        for w in range(0,width):
            f.write(" %.8e" % values[height-1-h,w]);
        f.write("\r\n")
    f.close()


# save 2d sections as GeoTIFF with fake coordinates and true aspect ratio
# ds2fakegtif(ds2d_plus.rotstd, 'ds2d_plus_rotstd.tif')
def da2fakegtif(data, filename):
    shape = data.shape
    pixelz = round(data.z.values[1]-data.z.values[0],5)
    pixell = round(data.l.values[1]-data.l.values[0],5)

    types = ['uint8','uint16','int16','uint32','int32','float32','float64']
    gtypes = [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
             gdal.GDT_Float32, gdal.GDT_Float64]

    dtype = data.values.dtype
    tidx = types.index(dtype)
    gtype = gtypes[tidx]
    if tidx in [0,1,2,3,4]:
        nodata = np.iinfo(dtype).min
    else:
        nodata = 170141000918780003225695629360656023552.000

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(filename, shape[1], shape[0], 1, gtype, options = [ 'COMPRESS=LZW' ])
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst.SetGeoTransform( [ 0, pixell, 0, max(data.z.values), 0, -pixelz ] )
    if data.epsg != '':
        srs = osr.SpatialReference()
        #srs.SetWellKnownGeogCS("WGS84")
        srs.ImportFromEPSG(int(data.epsg))
        dst.SetProjection( srs.ExportToWkt() )

    arr = np.nan*np.ones(data.values.shape)
    h = arr.shape[0]
    for z in range(0,h):
        arr[z,:] = data.values[h-z-1,:]
    arr[np.isnan(arr)] = nodata

    dst.GetRasterBand(1).SetNoDataValue(nodata)
    dst.GetRasterBand(1).WriteArray(arr)

# deprecated - use dem.interp_like(raster, 'linear') or dem.interp_like(raster, 'nearest') instead
def dem2df(dem, _df):
    import numpy as np
    df = _df.copy()
    # calculate x,y indicies on DEM raster
    xs = np.array((df.x - dem.ulx)/dem.xres, dtype=int)
    ys = np.array((df.y - dem.uly)/dem.yres, dtype=int)
    # ignore idices outside of DEM raster
    # get z values from DEM
    df['z'] = [dem.values[yidx,xidx] if (yidx>=0 and yidx<dem.shape[0] and xidx>=0 and xidx<dem.shape[1]) else 0
                        for (yidx,xidx) in zip(ys,xs)]
    return df

def ogrline2grid(raster,line):
    import numpy as np
    prev = None
    df = None
    for idx in range(len(line)):
        row = line.iloc[idx,:]
        x = row['x']
        y = row['y']
        # get pixel coordinates
        px = np.argmin(abs(raster.x.values-row['x']))
        py = np.argmin(abs(raster.y.values-row['y']))
        #print row
        #print idx, px, py
        if prev is not None:
            #print '\tcalculate segment...'
            if abs(px-prev[0]) >= abs(py-prev[1]):
                #print '\tdx > dy'
                maxlen = abs(prev[0]-px)+1
            else:
                #print '\tdy > dx'
                maxlen = abs(prev[1]-py)+1
            #xs = [int(round(x)) for x in np.linspace(prev[0],x,maxlen)]
            #ys = [int(round(y)) for y in np.linspace(prev[1],y,maxlen)]
            xs = np.linspace(prev[2],x,maxlen)
            ys = np.linspace(prev[3],y,maxlen)
            #print xs
            #print ys
            _df = pd.DataFrame.from_dict({'x':xs, 'y':ys})
            #print df.head()
            #print df.tail()
            if df is None:
                df = _df
            else:
                df = df.append([_df])
        prev = (px,py,x,y)
    df['z'] = 0
    return df

# save 2d sections as TXT files with real coordinates
def da2txt(da, filename):
    import numpy as np
    vals = da.values
    ls = da.l.values
    xs = da.x.values
    ys = da.y.values
    zs = da.z.values
    #print l,x,y,z

    with open(filename, "w") as f:
        f.write("x,y,z,%s\r\n" % da.name)
        for lidx, l in enumerate(ls):
            x = xs[lidx]
            y = ys[lidx]
            for zidx, z in enumerate(zs):
                z = zs[zidx]
                val = vals[zidx,lidx]
                #print x, y, z, val
                if np.isnan(val):
                    continue
                f.write("%.1f,%.1f,%.1f,%f\r\n" % (x, y, z, val));


def da2ascii(da, filename):
    import numpy as np
    types = ['uint8','uint16','int16','uint32','int32','float32','float64']
    dtype = da.values.dtype
    tidx = types.index(dtype)
    if tidx in [0,1,2,3,4]:
        nodata = np.iinfo(dtype).min
        nodata_str = "%d" % nodata
        pattern = " %d"
    else:
        nodata = 170141000918780003225695629360656023552.000
        nodata_str = "%f" % nodata
        pattern = " %.8e"

    minx = np.min(da.x.values)
    miny = np.min(da.y.values)

    pixelx = np.diff(da.x.values)[0]
    pixely = np.diff(da.y.values)[0]
    assert( abs(pixelx) == abs(pixely) )

    if pixely < 0:
        values = np.flipud(da.values)
    else:
        values = da.values

    height = da.shape[0]
    width  = da.shape[1]

    f = open(filename, 'w')

    f.write("ncols         %i\r\n" % width);
    f.write("nrows         %i\r\n" % height);
    f.write("xllcorner     %f\r\n" % (minx-pixelx/2));
    # TODO: CHECK FOR pixely > 0
    if pixely < 0:
        f.write("yllcorner     %f\r\n" % (miny+pixely/2));
    else:
        f.write("yllcorner     %f\r\n" % (miny-pixely/2));
    f.write("cellsize      %f\r\n" % pixelx);
    f.write("NODATA_value  %s\r\n" % nodata_str);
    for h in range(0,height):
        for w in range(0,width):
            f.write( pattern % values[height-1-h,w]);
        f.write("\r\n")
    f.close()

#q = [25,75]
def da_percentile(da, q):
    import numpy as np
    ds = da.copy(deep=True)
    pcnt = np.nanpercentile(da.values.reshape(-1),q)
    da.values = np.clip(da.values,pcnt[0],pcnt[1])
    return da

#https://stackoverflow.com/questions/11727822/reading-a-vtk-file-with-python

def vtk2da(filename, varname='None'):
    from vtk import vtkStructuredPointsReader
    from vtk.util import numpy_support as VN

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
def da2vtk1(da, filename, filter_by_output_range=None):
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
def ds2vtk3(ds, name, filename):
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

def da2vtk1_int(da, filename):
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

#https://stackoverflow.com/questions/39073973/how-to-generate-a-matrix-with-circle-of-ones-in-numpy-scipy
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

# create worldfile to define image coordinates
def worldfile_tofile(fname, area, dimensions):
    import os
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
    import ee
    import geopandas as gpd
    from shapely.ops import Polygon
    # WGS84 coordinates
    geom = Polygon(GEEimage.getInfo()['properties']['system:footprint']['coordinates'][0])
    # define 1st band projection
    proj = GEEimage.getInfo()['bands'][0]['crs']
    # extract area bounds in the 1st band projection
    area = gpd.GeoSeries(geom,crs='epsg:4326').to_crs(proj)[0].bounds
    GEEurl = GEEimage\
        .visualize(**vis)\
        .getThumbURL({'dimensions':dimensions, 'format': 'jpg'})
    #print (GEEurl)
    if fname is not None:
        geeurl_tofile(GEEurl, fname)
        worldfile_tofile(fname, area, dimensions)
    return {'url': GEEurl, 'width': dimensions[0], 'height': dimensions[1]}

def split_rect(rect, n):
    import numpy as np
    lats = np.linspace(rect[0], rect[2], n+1)
    lons = np.linspace(rect[1], rect[3], n+1)
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


def vtkpoints2ds(filename):
    import xarray as xr
    import numpy as np
    #from vtk import vtkStructuredGridReader
    from vtk import vtkStructuredPointsReader
    from vtk.util import numpy_support as VN

    reader = vtkStructuredPointsReader()
    reader.SetFileName(filename)
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


#writer.WriteToOutputStringOn()
#writer.Write()
#binary_string = writer.GetBinaryOutputString()
def vtkpoints2ds(filename_or_binarystring):
    import xarray as xr
    import numpy as np
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

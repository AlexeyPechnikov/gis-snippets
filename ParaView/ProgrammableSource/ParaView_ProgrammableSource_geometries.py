import vtk
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely as sp

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

# optional encoding
ENCODING = None
#ENCODING = 'utf-8'

# optional topography filename
DEM = None
# optional label field
#LABEL = 'NAME'
LABEL = None
# optional coordinate systems
DEFAULT_DEM_EPSG = 4326
DEFAULT_DF_EPSG = 4326

DF = "/Users/mbg/Documents/ne_50m_admin_0_countries_lakes/ne_50m_admin_0_countries_lakes.shp"
#DEM = "Users/mbg/Documents/GEBCO_2019/GEBCO_2019.nc"

if self.GetOutput().IsA("vtkMultiBlockDataSet"):
    print ("vtkMultiBlockDataSet output")
elif self.GetOutput().IsA("vtkPolyData"):
    print ("vtkAppendPolyData output")
else:
    print ("Unsupported output")
    return

# Load shapefile
df = gpd.read_file(DF, encoding=ENCODING)
if (len(df)) == 0:
    return
if LABEL is not None:
    if LABEL not in df.columns.values:
        print ("Label field is not exists: ", LABEL)
        print(df.columns.values)
        return
    else:
        df = df.sort_values(LABEL)

# TEST
#df = df[df.NAME=='United States of America']
#df = df.head(16)
print (len(df))

# load DEM
if DEM:
    dem = xr.open_rasterio(DEM)
    if 'crs' in dem.attrs.keys():
        epsg = int(dem.crs.split(':')[1])
    else:
        # DEM coordinate system is not defined, use WGS84 as default
        epsg = DEFAULT_DEM_EPSG
    print (epsg)
    # Reproject if needed
    if df.crs == {}:
        # vector coordinate system is not defined, use WGS84 as default
        df.crs = {'init': 'epsg:'+str(DEFAULT_DF_EPSG)}
    df['geometry'] = df['geometry'].to_crs(epsg=epsg)

if self.GetOutput().IsA("vtkMultiBlockDataSet"):
    mb = self.GetOutput()
    mb.SetNumberOfBlocks(len(df))
elif self.GetOutput().IsA("vtkPolyData"):
    output = vtk.vtkAppendPolyData()

for rowidx,row in df.reset_index().iterrows():
    shapes = vtk.vtkAppendPolyData()
    # split (multi)geometries to parts
    if row.geometry.geometryType()[:5] == 'Multi':
        geoms = row.geometry.geoms
    else:
        geoms = [row.geometry]
    # iterate parts of (multi)geometry
    for geom in geoms:
        if geom.type == 'Polygon':
            # use exterior coordinates only
            coords = geom.exterior.coords
        else:
            coords = geom.coords

        count = (len(coords))
        #print ("count",count)

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        if geom.type == 'Polygon':
            # for closed contoures only
            cells.InsertNextCell(count)
        elif geom.type == 'Point':
            # for point
            cells.InsertNextCell(count)
        else:
            # for segmented line only
            cells.InsertNextCell(count-1)

        for idx in range(count)[1 if count > 1 else 0:]:
            # Point and PointZ
            if len(coords[idx]) == 2:
                x0,y0 = coords[idx-1]
                x,y = coords[idx]
                # Z coordinate is not defined
                z = 0
            else:
                x0,y0,_z0 = coords[idx-1]
                x,y,z = coords[idx]
            if DEM:
                z = float(dem.sel(x=x,y=y,method='nearest'))
            pointId = points.InsertNextPoint(x, y, z)
            cells.InsertCellPoint(pointId)

        if geom.type == 'Polygon':
            # to close the contour
            cells.InsertCellPoint(0)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        if count > 1:
            polyData.SetLines(cells)
        else:
            polyData.SetVerts(cells)

        # merge all objects into group
        shapes.AddInputData(polyData)

    # compose vtkPolyData
    shapes.Update()
    polyData = shapes.GetOutput()

    for colname in df.columns:
        if colname == 'geometry':
            continue
        dtype = df[colname].dtype
        #print (colname,dtype)
        # define attribute as array
        if dtype in ['int64']:
            vtk_arr = vtk.vtkIntArray()
        elif dtype in ['float64']:
            vtk_arr = vtk.vtkFloatArray()
        elif dtype in ['bool']:
            vtk_arr = vtk.vtkBitArray()
        else:
            vtk_arr = vtk.vtkStringArray()
        vtk_arr.SetNumberOfComponents(1)
        vtk_arr.SetName(colname)
        val = row[colname]
        # some different datatypes could be saved as strings
        if isinstance(vtk_arr, vtk.vtkStringArray):
            val = val.encode('utf-8')
        for _ in range(polyData.GetNumberOfCells()):
            vtk_arr.InsertNextValue(val)
        polyData.GetCellData().AddArray(vtk_arr)

    if self.GetOutput().IsA("vtkMultiBlockDataSet"):
        # use index if label is not defined
        if LABEL:
            label = row[LABEL]
            if df[LABEL].dtype in ['O','str']:
                label = label.encode('utf-8')
            else:
                label=str(label)
            #print (label)
        else:
            label = str(rowidx)
        # add to multiblock dataset
        mb.SetBlock( rowidx, polyData )
        mb.GetMetaData( rowidx ).Set( vtk.vtkCompositeDataSet.NAME(), label)
    else:
        output.AddInputData(polyData)

if self.GetOutput().IsA("vtkPolyData"):
    output.Update()
    self.GetPolyDataOutput().ShallowCopy(output.GetOutput())

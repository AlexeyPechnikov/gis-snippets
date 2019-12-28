## Test data files
Test data files required:

[1] https://github.com/mobigroup/ParaView-MoshaFault/blob/master/data/GEBCO_2019/GEBCO_2019.subset.32639.0.5km.tif

[2] Damavand.shp see here https://github.com/mobigroup/ParaView-MoshaFault/tree/master/data/volcano

[3] data60.h5 generated as described in [3]

## vtkMultiblockDataSet

RequestInformation Script is not required for vtkMultiblockDataSet output.

### vtkMultiblockDataSet (read WGS84 volcano shapefile and EPSG:32639 topology geotif)

#### Script
```
import vtk
import xarray as xr
import geopandas as gpd

SHP = "/Users/mbg/Documents/volcano/Damavand.shp"
DEM="/Users/mbg/Documents/GEBCO_2019/GEBCO_2019.subset.32639.0.5km.tif"

dem = xr.open_rasterio(DEM)
epsg = int(dem.crs.split(':')[1])
print (epsg)

shp = gpd.read_file(SHP)
print (len(shp), shp.crs)

# Reproject if needed
if shp.crs == {}:
    # coordinate system is not defined, use WGS84 as default
    shp.crs = {'init': 'epsg:4326'}
shp['geometry'] = shp['geometry'].to_crs(epsg=epsg)

# get output tkMultiBlockDataSet()
print ("output",self.GetOutput().IsA("vtkMultiBlockDataSet"))
mb = self.GetOutput()
mb.SetNumberOfBlocks(len(shp))

for rowidx, row in shp.reset_index().iterrows():
    geom = row.geometry

    x = geom.x
    y = geom.y
    z = float(dem.sel(x=x,y=y,method='nearest'))

    # Create a polydata object
    point = vtk.vtkPolyData()

    # Create the geometry of a point (the coordinate)
    _points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    _vertices = vtk.vtkCellArray()
    
    id = _points.InsertNextPoint([x,y,z])
    _vertices.InsertNextCell(1)
    _vertices.InsertCellPoint(id)
    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(_points)
    point.SetVerts(_vertices)

    mb.SetBlock( rowidx, point )
    mb.GetMetaData( rowidx ).Set( vtk.vtkCompositeDataSet.NAME(), str(rowidx) )
```
![ParaView ProgrammableSource MultiblockDataSet](ParaView_ProgrammableSource_MultiblockDataSet.jpg)

## vtkPolyData

RequestInformation Script is not required for vtkPolyData output.

### vtkPolyData (read WGS84 volcano shapefile and EPSG:32639 topology geotif)

#### Script
```
import vtk
import xarray as xr
import geopandas as gpd

SHP = "/Users/mbg/Documents/volcano/Damavand.shp"
DEM="/Users/mbg/Documents/GEBCO_2019/GEBCO_2019.subset.32639.0.5km.tif"

dem = xr.open_rasterio(DEM)
epsg = int(dem.crs.split(':')[1])
print (epsg)

shp = gpd.read_file(SHP)
print (len(shp), shp.crs)

# Reproject if needed
if shp.crs == {}:
    # coordinate system is not defined, use WGS84 as default
    shp.crs = {'init': 'epsg:4326'}
shp['geometry'] = shp['geometry'].to_crs(epsg=epsg)

# Create a polydata object
point = vtk.vtkPolyData()

for rowidx, row in shp.reset_index().iterrows():
    geom = row.geometry

    x = geom.x
    y = geom.y
    z = float(dem.sel(x=x,y=y,method='nearest'))

    # Create the geometry of a point (the coordinate)
    _points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    _vertices = vtk.vtkCellArray()
    
    id = _points.InsertNextPoint([x,y,z])
    _vertices.InsertNextCell(1)
    _vertices.InsertCellPoint(id)
    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(_points)
    point.SetVerts(_vertices)

self.GetPolyDataOutput().ShallowCopy(point)
```
![ParaView ProgrammableSource PolyData](ParaView_ProgrammableSource_PolyData.jpg)

## vtkImageData

Generate simple HDF5 files as described by this link
https://blog.kitware.com/developing-hdf5-readers-using-vtkpythonalgorithm/

Add the scripts to textareas and turn off checkbox "Map scalars" and edit any script and click "Apply" (ignore all error messages) and turn on the checkbox again. Without this magic we can't see the output geometries!

#### Script (RequestInformation)
```
from paraview import util

pdi = self.GetOutput()
extent = pdi.GetExtent()
util.SetOutputWholeExtent(self, extent)
```

### vtkImageData (pre-defined data)

#### Script
```
import vtk
dx = 2.0
grid = vtk.vtkImageData()
grid.SetOrigin(0, 0, 0) # default values
grid.SetSpacing(dx, dx, dx)
grid.SetDimensions(5, 8, 10) # number of points in each direction
# print grid.GetNumberOfPoints()
# print grid.GetNumberOfCells()
array = vtk.vtkDoubleArray()
array.SetNumberOfComponents(1) # this is 3 for a vector
array.SetNumberOfTuples(grid.GetNumberOfPoints())
for i in range(grid.GetNumberOfPoints()):
    array.SetValue(i, i)

grid.GetPointData().AddArray(array)
# print grid.GetPointData().GetNumberOfArrays()
array.SetName("unit array")

self.GetImageDataOutput().ShallowCopy(grid)
```

![ParaView ProgrammableSource ImageData](ParaView_ProgrammableSource_ImageData.jpg)

### vtkImageData (read HDF5 data file)

#### Script
```
import vtk, h5py
from vtk.util import numpy_support

filename = "data60.h5"
f = h5py.File(filename, 'r')
data = f['RTData'][:]

dx = 1.0
grid = vtk.vtkImageData()
grid.SetOrigin(0, 0, 0) # default values
grid.SetSpacing(dx, dx, dx)
# Note that we flip the dimensions here because
# VTK's order is Fortran whereas h5py writes in
# C order.
grid.SetDimensions(data.shape[::-1]) # number of points in each direction

array = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT) 
grid.GetPointData().AddArray(array)
# print grid.GetPointData().GetNumberOfArrays()
array.SetName("RTData")

self.GetImageDataOutput().ShallowCopy(grid)
```
![ParaView ProgrammableSource ImageData](ParaView_ProgrammableSource_ImageData2.jpg)

#### Script works with output object directly
```
import vtk, h5py
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa

filename = "data60.h5"
f = h5py.File(filename, 'r')
data = f['RTData'][:]

dx = 1.0
grid = dsa.WrapDataObject(self.GetImageDataOutput())
grid.SetOrigin(0, 0, 0) # default values
grid.SetSpacing(dx, dx, dx)
# Note that we flip the dimensions here because
# VTK's order is Fortran whereas h5py writes in
# C order.
grid.SetDimensions(data.shape[::-1]) # number of points in each direction

array = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT) 
grid.GetPointData().AddArray(array)
# print grid.GetPointData().GetNumberOfArrays()
array.SetName("RTData")

grid.PointData.SetActiveScalars('RTData')
```
![ParaView ProgrammableSource ImageData](ParaView_ProgrammableSource_ImageData2.jpg)

## References

[1] https://www.paraview.org/Wiki/Python_Programmable_Filter

[2] https://stackoverflow.com/questions/7666981/how-to-set-data-values-on-a-vtkstructuredgrid

[3] https://blog.kitware.com/developing-hdf5-readers-using-vtkpythonalgorithm/

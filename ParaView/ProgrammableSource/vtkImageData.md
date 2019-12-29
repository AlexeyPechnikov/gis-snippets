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

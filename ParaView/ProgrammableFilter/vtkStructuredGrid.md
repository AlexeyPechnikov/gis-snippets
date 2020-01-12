## vtkStructuredGrid

### vtkStructuredGrid (median filter)

#### Script
```
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.ndimage.filters import median_filter

# prepare input data
print (inputs[0].GetDimensions())
vtk_arr = inputs[0].PointData['trace']
shape = inputs[0].GetDimensions()
arr = vtk_to_numpy(vtk_arr).reshape(shape[::-1])

# process
#arr[200:,:,:] = -10000
def unit_circle_2d(r):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt( A[:,None] + A)
    # circle
    #return (np.abs(dists-r)<=0).astype(int)
    # filled circle
    if r <= 2:
        return ((dists-r)<=0).astype(int)
    return ((dists-r)<=0.5).astype(int)
# z, y, x
r = 5
rz = 1
footprint = np.array((2*rz+1)*[unit_circle_2d(r)])
print (footprint.shape)
arr = median_filter(arr, footprint=footprint, mode='nearest')

# prepare output data
output.PointData.append(arr.ravel(), "arr")
```
![ParaView ProgrammableFilter StructuredGrid](ParaView_ProgrammableFilter_StructuredGrid.jpg)

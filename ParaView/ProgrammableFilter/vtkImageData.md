## vtkImageData to vtkStructuredGrid

Convert vtkImageData to vtkStructuredGrid. Set "Output Data Set Type" to vtkStructuredGrid.

```
from vtk.util import numpy_support as vn
import numpy as np
from vtk import vtkStructuredGrid, vtkPoints

pdi = self.GetInput()
pdo =  self.GetOutput()

_coords = []
for i in range(pdi.GetNumberOfPoints()):
    _coords.append(pdi.GetPoint(i))
coords = np.asarray(_coords)

vtk_points = vtkPoints()
points = np.column_stack((coords[:,0],coords[:,1],coords[:,2]))
_points = vn.numpy_to_vtk(points, deep=True)
vtk_points.SetData(_points)

sgrid = vtkStructuredGrid()
sgrid.SetDimensions(pdi.GetDimensions())
sgrid.SetPoints(vtk_points)

for idx in range(pdi.GetPointData().GetNumberOfArrays()):
    col = pdi.GetPointData().GetArrayName(idx)
    vtk_array = pdi.GetPointData().GetArray(idx)
    sgrid.GetPointData().AddArray(vtk_array)

pdo.ShallowCopy(sgrid)
```

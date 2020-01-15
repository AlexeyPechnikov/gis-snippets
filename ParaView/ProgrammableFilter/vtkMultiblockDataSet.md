## vtkMultiblockDataSet

Before we can launch the filter we need to create ParaView Table from EPSG:32614 drill locations CSV (2).

"RequestInformation Script" and "RequestUpdateExtent Script" are not required for vtkMultiblockDataSet output.

### vtkMultiblockDataSet (process EPSG:32614 drill locations from ParaView Table)

For better visialization turn on "Render Lines as Tubes" checkbox and set "Line Width" equal to 10.

#### Script
```
import vtk
import numpy as np
import pandas as pd
import math

vtk_table = self.GetTableInput()
# table headers
cols = []
for icol in range( vtk_table.GetNumberOfColumns() ):
    name = vtk_table.GetColumn(icol).GetName()
    cols.append( name )

values = []
for irow in range(vtk_table.GetNumberOfRows()):
    _values = []
    for icol in range( vtk_table.GetNumberOfColumns() ):
        val = vtk_table.GetColumn(icol).GetValue(irow)
        _values.append(val)
    values.append(_values)

df = pd.DataFrame.from_records(values, columns=cols)
#print (df)

# rename columns to easy use
df = df.rename(columns={'Easting': 'x','Northing':'y','Elevation':'z'})

# https://en.wikipedia.org/wiki/Spherical_coordinate_system
# Spherical coordinates (r, θ, φ) as often used in mathematics:
# radial distance r, azimuthal angle θ, and polar angle φ. 
df['theta'] = 1./2*math.pi - math.pi*df.Az/180
df['phi'] = math.pi*(90 - df.Dip)/180
# 1st point
df['dx'] = np.round(df.Length*np.sin(df.phi)*np.cos(df.theta))
df['dy'] = np.round(df.Length*np.sin(df.phi)*np.sin(df.theta))
df['dz'] = np.round(df.Length*np.cos(df.phi))
# 2nd point
df['x2'] = df.x + df.dx
df['y2'] = df.y + df.dy
df['z2'] = df.z + df.dz
# label
df['label'] = df.Hole_ID + ' [' + df.Zone + ']'

# get output tkMultiBlockDataSet()
print ("output",self.GetOutput().IsA("vtkMultiBlockDataSet"))
mb = self.GetOutput()
mb.SetNumberOfBlocks(len(df))

for idx,well in df.iterrows():
    #print (idx,well)
    
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(2)
    points.SetPoint(0, well['x'],well['y'],well['z'])
    points.SetPoint(1, well['x2'],well['y2'],well['z2'])
    
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(2)
    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)
    
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    time = vtk.vtkFloatArray()
    time.SetNumberOfComponents(1)
    time.SetName("year");
    time.InsertNextValue(well.Year)
    polyData.GetCellData().AddArray(time)
    
    zone = vtk.vtkStringArray()
    zone.SetNumberOfComponents(1)
    zone.SetName("zone")
    zone.InsertNextValue(well.Zone)
    polyData.GetCellData().AddArray(zone)

    wtype = vtk.vtkStringArray()
    wtype.SetNumberOfComponents(1)
    wtype.SetName("type")
    wtype.InsertNextValue(well.Type)
    polyData.GetCellData().AddArray(wtype)
    
    # add to multiblock dataset
    mb.SetBlock( idx, polyData )
    mb.GetMetaData( idx ).Set( vtk.vtkCompositeDataSet.NAME(), well['label'] )
```
![ParaView ProgrammableFilter MultiblockDataSet](ParaView_ProgrammableFilter_MultiblockDataSet.jpg)

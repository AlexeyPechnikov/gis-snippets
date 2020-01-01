## vtkPolyData from vtkMultiblockDataSet

Extract one block from vtkMultiblockDataSet as vtkPolyData. For practical tasks that's better to use ExtractBlock->MergeBlocks ParaView filters instead.

```
BLOCKIDX = 0
pdi = self.GetInputDataObject(0,BLOCKIDX)
# vtkPolyData
block0 = pdi.GetBlock(0)
pdo = self.GetOutputDataObject(0)
pdo.ShallowCopy(block0)
```

## How to load data into ParaView using it's ProgrammableFilter feature

There are some different ways to enhance ParaView features by custom Python code:

1. Use Python console (the best way for one-time commands which are outside of the processing pipeline),
2. Use [Programmable Source](../ProgrammableSource/README.md) (the best way for custom code without pipeline input and the complete processing pipeline),
3. Use [Programmable Filter](../ProgrammableFilter/README.md) (the best way for custom code with pipeline input and the complete processing pipeline),
4. Use Plugins (the best way for well formalized code modules).

Below use explaining the way #3. In this case we use Python code integrated into ParaView processing pipeline with dependencies to other pipeline objects. Way #2 is very similar but it's not depends of other pipeline items.

[Install Python modules](../install.md)

[Prepare data files](../datafiles.md)

[ProgrammableFilter vtkMultiblockDataSet](vtkMultiblockDataSet.md)

[References](../references.md)

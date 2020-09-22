# AutoLM

## Intruduction
This Package is designed to help automate the landmarking process of 2D and 3D image data. The package
includes a range of data augmentation and preparation functions that take care of the Image data, as well as the landmark data. It also includes some auxiliary functionality to handle large datasets and training on the GPU. Finally, the included
procrustes alignment function can be used to estimate the prediction accuracy on a case by case basis.


## Tutorial
A typical workflow with a 3D dataset can be found in the [demo file](../AutoLM.jl/demo.jl). Most functions work with either 2D or 3D data. If not, the package usually includes separate functions to handle either 2D or 3D data. The intended use case is always made clear in the name of the function. Detailed documentation is included with every function.

# AutoLandmarking

## Introduction
This package is designed to help automate the landmarking process of 2D and 3D image data. It includes a range of data normalization, augmentation and preparation functions that take care of the image data, as well as the corresponding landmark data. This package includes some auxiliary functionality to handle large datasets and training on the GPU. Finally, the included procrustes alignment function can be used to estimate the prediction accuracy on a case by case basis.


## Tutorial
A typical workflow with a 3D dataset can be found in the [demo file](../master/demo.jl). Most functions work with either 2D or 3D data. If not, the package usually includes separate functions to handle either 2D or 3D data. The intended use case is always made clear in the name of the function. Detailed documentation is included with every function. The following image depicts a typical data pipeline for landmarked 2D and 3D images:
![alt text](https://github.com/imi-diener/AutoLandmarking.jl/blob/master/test/images/data_flow.jpg "Data flow")


## Landmark format
All the functions are designed to work with the landmarking format used by the Avizo software. A short example: say we have a volume with the size 128x128x128 voxels and we have a landmark at the location (x=40, y=55, z=32) with each coordinate being the number of voxel lengths in that dimension; then in this case the landmarks would be x=4.0, y=5.5, z=3.2. In short, coordinates are the number of voxel lengths from the origin (0,0,0) in any dimension divided by 10. Any other format will not work with most of the functions.

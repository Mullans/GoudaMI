# GoudaMI
These are all of my common utilities for different medical imaging projects. While they were mostly written while intended for use with 3D images, almost everything should work just as well with 2D or n-D images too.


## Requirements:
* [numpy](https://numpy.org/)
* [SimpleITK](https://simpleitk.org/)

### Optional Requirements
* [itk](https://itk.org/) - Used as either an interface for vtk or as an optional backend for SmartImage objects
* [scipy](scipy.org) - Used in some [older project-specific methods](GoudaMI/project_specific.py)
* [SuperElastix](https://github.com/SuperElastix) - Used mostly in [registration](GoudaMI/registration.py)
* [vtk](https://vtk.org/) - Used for some [surface smoothing](GoudaMI/vtk_utils.py) and [data conversion](GoudaMI/convert.py)


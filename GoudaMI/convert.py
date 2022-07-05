from functools import wraps
import os

import itk
import numpy as np
import SimpleITK as sitk
import vtk

from .smart_image import SmartImage
#TODO - create wrap4XXX for itk, sitk, and vtk to take any format, convert to chosen, and return as original format


def wrap4itk(func):
    """Wrap a method that takes a vtk.ImageData and returns a vtk.PolyData"""
    @wraps(func)
    def wrapped_func(image, *args, **kwargs):
        input_type = ''
        input_direction = image.GetDirection()
        if isinstance(image, SmartImage):
            input_type = 'Smart,' + image.default_type
            image = image.itk_image
        elif isinstance(image, sitk.Image):
            image = sitk2itk(image)
            input_type = 'sitk'
        elif isinstance(image, itk.Image):
            input_type = 'itk'
        else:
            raise ValueError("Unknown input type: {}".format(type(image)))
        image = itk.vtk_image_from_image(image)
        result = func(image, *args, **kwargs)

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(result)
        stencil.SetInformationInput(image)
        stencil.Update()

        converter = vtk.vtkImageStencilToImage()
        converter.SetInputData(stencil.GetOutput())
        converter.SetInsideValue(1)
        converter.SetOutsideValue(0)
        converter.SetOutputScalarTypeToUnsignedChar()
        converter.Update()

        result = itk.image_from_vtk_image(converter.GetOutput())
        #FIXME - result may have different direction from input image
        if input_type.startswith('Smart'):
            _, default_type = input_type.split(',')
            result = SmartImage(result, default_type=default_type)
        elif input_type == 'sitk':
            result = itk2sitk(result)
        else:
            # Should be input_type == 'itk'
            pass
        result.SetDirection(input_direction)
        return result
    return wrapped_func


def sitk2vtk(sitk_pointer, nb_points=None):
    import vtk.util.numpy_support
    numpy_array = sitk.GetArrayFromImage(sitk_pointer)
    size = list(sitk_pointer.GetSize())
    origin = list(sitk_pointer.GetOrigin())
    spacing = list(sitk_pointer.GetSpacing())
    label = vtk.util.numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Convert the VTK array to vtkImageData
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(size)
    img_vtk.SetSpacing(spacing)
    img_vtk.SetOrigin(origin)
    img_vtk.GetPointData().SetScalars(label)

    cube_filter = vtk.vtkDiscreteMarchingCubes()
    cube_filter.SetInputData(img_vtk)
    cube_filter.GenerateValues(1, 1, 1)
    cube_filter.Update()

    return cube_filter.GetOutput()


def image2vtk(image):
    if isinstance(image, SmartImage):
        image = image.itk_image
    elif isinstance(image, sitk.Image):
        image = sitk2itk(image)
    return itk.vtk_image_from_image(image)


def vtk2image(image):
    return itk.image_from_vtk_image(image)


def sitk2itk(image):
    if isinstance(image, itk.Image):
        return image
    itk_image = itk.GetImageFromArray(sitk.GetArrayViewFromImage(image), is_vector=image.GetNumberOfComponentsPerPixel() > 1)
    itk_image.SetOrigin(image.GetOrigin())
    itk_image.SetSpacing(image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [image.GetDimension()] * 2)))
    return itk_image


def itk2sitk(image):
    if isinstance(image, sitk.Image):
        return image
    sitk_image = sitk.GetImageFromArray(itk.GetArrayViewFromImage(image), isVector=image.GetNumberOfComponentsPerPixel() > 1)
    sitk_image.SetOrigin(tuple(image.GetOrigin()))
    sitk_image.SetSpacing(tuple(image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return sitk_image


def images2series(images):
    slices = []
    for image in images:
        if isinstance(image, sitk.Image):
            slices.append(image)
        elif isinstance(image, itk.Image):
            slices.append(itk2sitk(image))
        elif isinstance(image, (str, os.PathLike)):
            slices.append(sitk.ReadImage(str(image)))
        else:
            raise ValueError('Unknown image type: {}'.format(type(image)))
    series = sitk.JoinSeries(slices)
    return series

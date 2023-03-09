import functools
import os
import shutil
import subprocess
import sys
import warnings
from typing import Optional

import gouda
import numpy as np
import SimpleITK as sitk

from GoudaMI.optional_imports import itk, vtk
from GoudaMI.smart_image import SmartImage, get_image_type, as_image, as_image_type, ImageType

# TODO - create wrap4XXX for itk, sitk, and vtk to take any format, convert to chosen, and return as original format


def as_array(image: ImageType) -> np.ndarray:
    image_type = get_image_type(image)
    if image_type == 'sitk':
        image = sitk.GetArrayFromImage(image)
    elif image_type == 'itk':
        image = itk.GetArrayFromImage(image)
    elif image_type == 'smartimage':
        image = image.as_array()
    else:
        image = np.asarray(image)
    return image


def as_view(image: ImageType) -> np.ndarray:
    image_type = get_image_type(image)
    if image_type == 'sitk':
        image = sitk.GetArrayViewFromImage(image)
    elif image_type == 'itk':
        image = itk.GetArrayViewFromImage(image)
    elif image_type == 'smartimage':
        image = image.as_view()
    else:
        image = np.asarray(image)
    return image


def wrap4itk(func):
    """Wrap a method that takes a vtk.ImageData and returns a vtk.PolyData"""
    @functools.wraps(func)
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


def wrap_numpy2numpy(func):
    """Wrap a method that takes a numpy.ndarray and returns a numpy.ndarray so that it takes any image and returns the same type"""
    @functools.wraps(func)
    def wrapped_func(image, *args, **kwargs):
        return_type = ''
        if isinstance(image, SmartImage):
            image = image.image
            return_type = 'smart'
        if isinstance(image, sitk.Image):
            arr = sitk.GetArrayViewFromImage(image)
            return_type += 'sitk'
        elif isinstance(image, itk.Image):
            arr = itk.GetArrayViewFromImage(image)
            return_type += 'itk'
        elif isinstance(image, np.ndarray):
            return_type = 'array'
        else:
            raise ValueError('Unknown image type: {}'.format(type(image)))

        arr = func(arr, *args, **kwargs)

        if return_type.endswith('sitk'):
            return_image = sitk.GetImageFromArray(arr)
            return_image.CopyInformation(image)
        elif return_type.endswith('itk'):
            return_image = itk.GetImageFromArray(arr)
            return_image.SetOrigin(image.GetOrigin())
            return_image.SetDirection(image.GetDirection())
            return_image.SetSpacing(image.GetSpacing())
        if return_type.startswith('smart'):
            return_image = SmartImage(return_image, default_type=return_type[5:])

        return return_image
    return wrapped_func


def wrap_sitk(func):
    """Wrap a method that takes SimpleITK.Image(s) and returns anything so that it takes any image and returns the method return type"""
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        new_args = []
        return_type = None
        for item in args:
            if isinstance(item, (itk.Image, sitk.Image, SmartImage)):
                if return_type is None:
                    return_type = get_image_type(item)
                new_args.append(as_image(item).sitk_image)
            else:
                new_args.append(item)
        new_kwargs = {}
        for key, item in kwargs.items():
            if isinstance(item, (itk.Image, sitk.Image, SmartImage)):
                new_kwargs[key] = as_image(item).sitk_image
            else:
                new_kwargs[key] = item
        result = func(*new_args, **new_kwargs)
        if gouda.is_iter(result, non_iter=(str, bytes, itk.Image, sitk.Image, SmartImage)):
            new_results = []
            for item in result:
                if isinstance(item, (itk.Image, sitk.Image, SmartImage)):
                    new_results.append(as_image_type(item, return_type))
            return tuple(new_results)
        elif isinstance(result, (itk.Image, sitk.Image, SmartImage)):
            return as_image_type(result, return_type)
        else:
            return result
    return wrapped_func


def convert_sitk(image):
    return_type = ''
    if isinstance(image, SmartImage):
        image = image.image
        return_type = 'smart'
    if isinstance(image, sitk.Image):
        return_type += 'sitk'
    elif isinstance(image, itk.Image):
        image = itk2sitk(image)
        return_type += 'itk'
    elif isinstance(image, np.ndarray):
        warnings.warn('Treating an array as an image ignores any physical parameters (origin, spacing, etc).')
        image = sitk.GetImageFromArray(image)
        return_type += 'array'
    else:
        raise ValueError('Unknown image type: {}'.format(type(image)))
    return image, return_type


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


def plastimatch_rt_to_nifti(rt_path: os.PathLike, dicom_path: os.PathLike, dst_path: os.PathLike, image_dst_path: Optional[os.PathLike]=None, post_check: bool=False, verbose: bool=True) -> str:
    """Convert an rt_struct file to a nifti file using plastimatch

    Parameters
    ----------
    rt_path: str
        Filepath to the structure object
    dicom_path: str
        Filepath to the corresponding reference dicom image directory
    dst_path: str
        Filepath to save the resulting nifti object
    image_dst_path: str | None
        Filepath to save a nifti version of the dicom reference image
    post_check: bool
        Whether to do a post-processing check to ensure single channel labels (the default is False)
    verbose: bool
        Whether to print warnings during execution (the default is True)
    Returns
    -------
    result_string: str
        The stdout from the plastimatch call
    """
    plastimatch_path = shutil.which('plastimatch')
    if plastimatch_path is None:
        raise ImportError('Could not find the plastimatch executable - see http://plastimatch.org/getting_started.html for installation help')

    if not os.path.exists(dicom_path):
        raise ValueError('Could not find reference dicom at: {}'.format(dicom_path))
    # print(f"plastimatch convert --input {rt_path} --output-ss-img {dst_path} --referenced-ct {dicom_path}")
    result = subprocess.run([
        'plastimatch', 'convert',
        '--input', rt_path,
        '--output-ss-img', dst_path,
        '--referenced-ct', dicom_path,
        # '--output-img', dst_path.replace('_contour', '_image')
    ], capture_output=True)
    result_string = result.stdout.decode()
    if not result_string.strip().endswith('Finished!'):
        warnings.warn('Process finished with issues - see stderr\n' + '-' * 41)
        print(result_string, file=sys.stderr)
        return
    elif 'Setting PIH from RDD' not in result_string:
        if verbose:
            warnings.warn('rt_struct converted without referencing dicom - see result string for details')

    if post_check:
        dst_path = str(dst_path)
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(dst_path)
        file_reader.ReadImageInformation()
        num_comp = file_reader.GetNumberOfComponents()
        if num_comp > 1:
            if verbose:
                warnings.warn(f'Contour saved as vector with {num_comp} components - converting into union vector label')
            label_img = sitk.ReadImage(dst_path)
            split_img = [sitk.VectorIndexSelectionCast(label_img, idx) for idx in range(num_comp)]
            union_label = functools.reduce(sitk.Or, split_img)
            # sitk.WriteImage(union_label, dst_path.replace('.nii.gz', '_union.nii.gz'))
            sitk.WriteImage(union_label, dst_path)  # should just overwrite the vector version

    if image_dst_path is not None:
        if os.path.exists(image_dst_path):
            if verbose:
                warnings.warn(f'File already exists at {image_dst_path} - skipping...')
        else:
            result2 = subprocess.run([
                'plastimatch', 'convert',
                '--input', dicom_path,
                '--output-img', image_dst_path,
                '--output-type', 'short'  # save it as int16
            ], capture_output=True)
            result_string2 = result2.stdout.decode()
            if not result_string2.strip().endswith('Finished!'):
                warnings.warn('Dicom image conversion finished with issues - see stdeerr\n' + '-' * 57)
                print(result_string2, file=sys.stderr)
    return result_string


def polydata_to_point_mask(polydata, ref_image):
    """Convert a vtkPolyData object to a binary mask where each point is a 1

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        The PolyData object to convert
    ref_image : ImageType
        Any reference object to use for size and spacing

    Returns
    -------
    SmartImage
        The resulting binary mask
    """
    src_points = vtk.util.numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    empty_arr = np.zeros(ref_image.GetSize()[::-1])
    phys_points = np.round(src_points / ref_image.GetSpacing()).astype(int)
    for item in phys_points:
        empty_arr[item[2], item[1], item[0]] = 1
    check_img = SmartImage(empty_arr)
    check_img.CopyInformation(ref_image)
    return check_img


def polydata_to_label_mask(polydata, ref_image):
    """Convert a vtkPolyData object to a binary mask where the PolyData is converted to a closed surface object

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        The PolyData object to convert
    ref_image : ImageType
        Any reference object to use for size and spacing

    Returns
    -------
    SmartImage
        The resulting binary mask
    """
    ref_image = as_image(ref_image)
    vtk_image = itk.vtk_image_from_image(ref_image.itk_image)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(polydata)
    stencil.SetOutputSpacing(ref_image.GetSpacing())
    stencil.SetOutputWholeExtent(vtk_image.GetExtent())
    stencil.Update()

    converter = vtk.vtkImageStencilToImage()
    converter.SetInputData(stencil.GetOutput())
    converter.SetInsideValue(1)
    converter.SetOutsideValue(0)
    converter.SetOutputScalarTypeToUnsignedChar()
    converter.Update()

    result = itk.image_from_vtk_image(converter.GetOutput())
    result = as_image(result)
    result.CopyInformation(ref_image)
    return result

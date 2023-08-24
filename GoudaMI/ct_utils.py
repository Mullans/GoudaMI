import glob
import functools
import os
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import gouda
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from GoudaMI import io
from GoudaMI.constants import MIN_INTENSITY_PULMONARY_CT
from GoudaMI.convert import as_view, wrap_numpy2numpy, wrap_image_func
from GoudaMI.optional_imports import itk
from GoudaMI.smart_image import (ImageRefType, ImageType, SmartImage, as_image, as_image_type, get_image_type, zeros_like)
from GoudaMI.smart_type import SmartType

# NOTE - reference for some interfacing: https://github.com/SimpleITK/SimpleITK/blob/4aabd77bddf508c1d55519fbf6002180a08f9208/Wrapping/Python/Python.i#L764-L794


def wrap_bounds(fn=None, padding=0, bg_val=0):
    """Wraps a method so that it only operates on the non-zero bounds of an image and re-pads it afterwards."""
    def internal_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(image, *args, **kwargs):
            image_type = get_image_type(image)
            image = as_image(image)
            bounds = get_bounds(image, bg_val=bg_val)[1]
            bound_slice = pad_bounds(bounds, padding, as_slice=True)
            result = func(image[bound_slice].sitk_image, *args, **kwargs)
            if not isinstance(result, (sitk.Image, SmartImage, np.ndarray)):
                return result

            low_padding = [int(item[0]) for item in bounds]
            upper_padding = [int(side - item[1]) for side, item in zip(image.GetSize(), bounds)]
            result = as_image(result)
            result = result.apply(sitk.ConstantPad, low_padding, upper_padding, bg_val)
            return as_image_type(result, image_type)
        return wrapped_func
    if fn is not None:
        return internal_wrapper(fn)
    return internal_wrapper


def get_bounds_slicers(image, padding=0, bg_val=0):
    bounds = get_bounds(image, bg_val=bg_val)[1]
    bounds_slice = pad_bounds(bounds, padding, image_size=image.GetSize(), as_slice=True)

    def crop_func(crop_image):
        return crop_image[bounds_slice]

    low_padding = [int(item.start) for item in bounds_slice]
    upper_padding = [int(side - item.stop) for side, item in zip(image.GetSize(), bounds_slice)]

    def uncrop_func(uncrop_image):
        if isinstance(uncrop_image, sitk.Image):
            return sitk.ConstantPad(uncrop_image, low_padding, upper_padding, bg_val)
        elif isinstance(uncrop_image, SmartImage):
            return uncrop_image.apply(sitk.ConstantPad, low_padding, upper_padding, bg_val)
        else:
            raise ValueError('Only sitk.Image and SmartImage are supported for get_bounds_slicers.')
    return crop_func, uncrop_func


def clip_image(image: ImageType, low: float, high: float):
    """Clip the intensity values in the image to a given range.
    """
    image_type = get_image_type(image)
    if image_type == 'smartimage':
        return image.window(min=low, max=high)
    elif image_type == 'sitk':
        return sitk.IntensityWindowing(image, low, high, low, high)
    elif image_type == 'itk':
        filter = itk.IntensityWindowingImageFilter.New(image)
        filter.SetInput(image)
        filter.SetOutputMaximum(max)
        filter.SetOutputMinimum(min)
        filter.SetWindowMaximum(max)
        filter.SetWindowMinimum(min)
        filter.Update()
        result = filter.GetOutput()
        return result
    else:
        raise TypeError('Unknown image type: {}'.format(type(image)))


@wrap_image_func('sitk')
def quick_open(img, radius=3, kernel=sitk.sitkBall):
    """Shortcut method for applying the sitk BinaryMorphologicalOpeningImageFilter"""
    crop_func, uncrop_func = get_bounds_slicers(img, padding=radius + 1)
    img = crop_func(img)
    opener = sitk.BinaryMorphologicalOpeningImageFilter()
    opener.SetKernelType(kernel)
    opener.SetKernelRadius(radius)
    opener.SetForegroundValue(1)
    opener.SetBackgroundValue(0)
    result = opener.Execute(img)
    return uncrop_func(result)


@wrap_image_func('sitk')
def quick_close(img, radius=3, kernel=sitk.sitkBall):
    """Shortcut method for applying the sitk BinaryMorphologicalClosingImageFilter"""
    crop_func, uncrop_func = get_bounds_slicers(img, padding=radius + 1)
    img = crop_func(img)
    closer = sitk.BinaryMorphologicalClosingImageFilter()
    closer.SetKernelType(kernel)
    closer.SetKernelRadius(radius)
    closer.SetForegroundValue(1)
    result = closer.Execute(img)
    return uncrop_func(result)


@wrap_image_func('sitk')
def quick_dilate(img, radius=3, kernel=sitk.sitkBall):
    """Shortcut method for applying the sitk BinaryDilateImageFilter"""
    crop_func, uncrop_func = get_bounds_slicers(img, padding=radius + 1)
    img = crop_func(img)
    dil_filter = sitk.BinaryDilateImageFilter()
    dil_filter.SetKernelType(kernel)
    dil_filter.SetKernelRadius(radius)
    dil_filter.SetForegroundValue(1)
    dil_filter.SetBackgroundValue(0)
    result = dil_filter.Execute(img)
    return uncrop_func(result)


@wrap_image_func('sitk')
def quick_erode(img, radius=3, kernel=sitk.sitkBall):
    """Shortcut for applying the sitk BinaryErodeImageFilter"""
    crop_func, uncrop_func = get_bounds_slicers(img, padding=radius + 1)
    img = crop_func(img)
    dil_filter = sitk.BinaryErodeImageFilter()
    dil_filter.SetKernelType(kernel)
    dil_filter.SetKernelRadius(radius)
    dil_filter.SetForegroundValue(1)
    dil_filter.SetBackgroundValue(0)
    result = dil_filter.Execute(img)
    return uncrop_func(result)


@wrap_image_func('sitk')
def quick_median(img, radius=3):
    crop_func, uncrop_func = get_bounds_slicers(img, padding=radius + 1)
    img = crop_func(img)
    dil_filter = sitk.BinaryMedianImageFilter()
    dil_filter.SetRadius(radius)
    dil_filter.SetForegroundValue(1)
    dil_filter.SetBackgroundValue(0)
    result = dil_filter.Execute(img)
    return uncrop_func(result)


def mask_body(image, resample=True):
    # Note - rounding error may cut off the top slice during resampling
    image_type = get_image_type(image)
    sampling_info = get_sampling_info(image)
    if resample:
        image = resample_iso_by_slice_size(image, 128, interp=sitk.sitkLinear)
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(3)
    image = median_filter.Execute(image)
    bin_img = sitk.BinaryThreshold(image, -500, 10000, 1, 0)
    labels = sitk.ConnectedComponent(bin_img)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(labels)
    body_label = [-1, -1]
    for label in lfilter.GetLabels():
        label_area = lfilter.GetNumberOfPixels(label)
        if label_area > body_label[1]:
            body_label = [label, label_area]
    bin_img = sitk.Equal(labels, body_label[0])
    bin_img = sitk.ConstantPad(bin_img, [0, 0, 1], [0, 0, 1], 1)
    bin_img = sitk.BinaryFillhole(bin_img)
    bin_img = sitk.Crop(bin_img, [0, 0, 1], [0, 0, 1])
    if resample:
        bin_img = resample_to_ref(bin_img, sampling_info, interp=sitk.sitkNearestNeighbor, outside_val=0)
    if image_type != 'sitk':
        bin_img = as_image(bin_img)
        if image_type == 'numpy':
            bin_img = bin_img.as_array()
        elif image_type == 'itk':
            bin_img = bin_img.itk_image
    return bin_img


def crop_image_to_mask(image: ImageType, mask: Optional[Union[ImageType, np.ndarray]] = None, crop_z=False, crop_quantile=50, return_bounds=False):
    if mask is None:
        mask = mask_body(image)
    mask = as_view(mask)

    front_view = mask.max(axis=1)
    side_view = mask.max(axis=2)
    x_starts = []
    x_stops = []
    y_starts = []
    y_stops = []
    for idx in range(front_view.shape[0]):
        xrow = front_view[idx]
        if xrow.sum() == 0:
            continue
        xstart, xstop = np.where(xrow > 0)[0][[0, -1]]
        x_starts.append(xstart)
        x_stops.append(xstop)

        yrow = side_view[idx]
        if yrow.sum() == 0:
            continue
        ystart, ystop = np.where(yrow > 0)[0][[0, -1]]
        y_starts.append(ystart)
        y_stops.append(ystop)

    if crop_z:
        column = mask.max(axis=(1, 2))
        zstart, zstop = np.where(column > 0)[0][[0, -1]]
        z_slice = slice(zstart, zstop)

    sizex, sizey, _ = image.GetSize()
    sizex = int(sizex)
    sizey = int(sizey)
    x_stop = sizex - np.percentile(x_starts, crop_quantile)
    x_start = sizex - np.percentile(x_stops, 100 - crop_quantile)
    y_stop = np.percentile(y_stops, 100 - crop_quantile)
    y_start = np.percentile(y_starts, crop_quantile)
    lengthx = x_stop - x_start
    lengthy = y_stop - y_start
    cropx = sizex - lengthx
    cropy = sizey - lengthy
    to_crop = min(cropx, cropy)

    left = sizex - x_stop
    right = x_start
    if right > left:
        diff = right - left
        extra = to_crop - diff
        x_slice = slice(int(extra / 2), sizex - int(diff + extra / 2))
    else:
        diff = left - right
        extra = to_crop - diff
        x_slice = slice(int(diff + extra / 2), sizex - int(extra / 2))

    front = y_start
    back = sizey - y_stop
    if front > back:
        diff = min(to_crop, front - back)
        extra = to_crop - diff
        y_slice = slice(int(diff + extra / 2), sizey - int(extra / 2))
    else:
        diff = min(to_crop, back - front)
        extra = to_crop - diff
        y_slice = slice(int(extra / 2), sizey - int(diff + extra / 2))

    if crop_z:
        bounds = (x_slice, y_slice, z_slice)
    else:
        bounds = (x_slice, y_slice)

    if return_bounds:
        return bounds

    cropped = image[bounds]
    return cropped


def resample_iso_by_slice_size(image, output_size, outside_val=MIN_INTENSITY_PULMONARY_CT, interp=sitk.sitkBSpline, presmooth=False):
    """Resample an image to isotropic spacing based on a given slice size"""
    if presmooth:
        image = sitk.RecursiveGaussian(image, 3)

    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    input_origin = image.GetOrigin()
    input_direction = image.GetDirection()
    if not hasattr(output_size, '__iter__'):  # single number
        output_size = [output_size, output_size, input_size[2]]
    elif len(output_size) == 2:  # just the x/y size
        output_size = list(output_size) + [input_size[2]]

    output_spacing = (np.array(input_size) * np.array(input_spacing)) / np.array(output_size)
    output_spacing[2] = output_spacing[0]
    output_size[2] = (input_size[2] * input_spacing[2]) / output_spacing[2]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interp)
    resample_filter.SetOutputDirection(input_direction)
    resample_filter.SetOutputOrigin(input_origin)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetSize([int(s) for s in output_size])
    resample_filter.SetDefaultPixelValue(outside_val)
    resampled_image = resample_filter.Execute(image)
    return resampled_image


def resample_to_ref(image: ImageType, ref: ImageRefType, outside_val: float = 0, interp: int = sitk.sitkNearestNeighbor) -> ImageType:
    """Resample an image to match a reference image

    Parameters
    ----------
    image : ImageType
        The image to resample
    ref : ImageRefType
        The reference to get physical parameters from
    outside_val : float, optional
        The outside value of the image, by default 0
    interp : int, optional
        The interpolator to use, by default sitk.sitkNearestNeighbor

    Raises
    ------
    ValueError
        Raised if the reference object is not an ImageType, dict, or SimpleITK.ImageFileReader
    """
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetInterpolator(interp)
    resampleFilter.SetDefaultPixelValue(outside_val)
    if isinstance(ref, itk.Image):
        ref = as_image(ref)
    if isinstance(ref, sitk.Image):
        resampleFilter.SetReferenceImage(ref)
    elif isinstance(ref, (sitk.ImageFileReader)):
        resampleFilter.SetSize(ref.GetSize())
        resampleFilter.SetOutputOrigin(ref.GetOrigin())
        resampleFilter.SetOutputSpacing(ref.GetSpacing())
        resampleFilter.SetOutputDirection(ref.GetDirection())
    elif isinstance(ref, dict):
        resampleFilter.SetSize(ref['size'])
        resampleFilter.SetOutputOrigin(ref['origin'])
        resampleFilter.SetOutputSpacing(ref['spacing'])
        resampleFilter.SetOutputDirection(ref['direction'])
    elif isinstance(ref, SmartImage):
        resampleFilter.SetSize(ref.GetSize().tolist())
        resampleFilter.SetOutputOrigin(ref.GetOrigin())
        resampleFilter.SetOutputSpacing(ref.GetSpacing())
        resampleFilter.SetOutputDirection(ref.GetDirection())
    else:
        raise ValueError("Unknown reference type: '{}'".format(type(ref)))
    resampleIm = resampleFilter.Execute(image)
    return resampleIm


def resample_separate_z(image, target_spacing, slice_interp=sitk.sitkBSpline, z_interp=sitk.sitkNearestNeighbor):
    """Resample an image to a target spacing, but use different z-interpolation"""
    if slice_interp < 2:
        slice_interp += 1
    if z_interp < 2:
        z_interp += 1
    # Some applications have 0=NN, 1=Linear, but sitk has 1=NN and 2=Linear
    new_shape = np.round((np.float32(image.GetSpacing()) / np.float32(target_spacing)) * np.float32(image.GetSize())).astype(int)

    slice_spacing = target_spacing[:2]
    slice_shape = new_shape[:2]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(slice_interp)
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetOutputSpacing(slice_spacing)
    resampler.SetOutputOrigin(image.GetOrigin()[:2])
    resampler.SetSize(slice_shape.tolist())
    mid_image = sitk.JoinSeries([resampler.Execute(image[:, :, idx]) for idx in range(image.GetSize()[-1])])
    mid_image.SetDirection(image.GetDirection())
    mid_image.SetOrigin(image.GetOrigin())
    mid_image.SetSpacing(np.concatenate([slice_spacing, [image.GetSpacing()[-1]]]))
    # ~2x Faster than a 3D resampling with only slice spacing changed

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(z_interp)
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(target_spacing)
    # Need this, otherwise top slice may be default value
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetSize(new_shape.tolist())

    result = resampler.Execute(mid_image)
    return result


def get_sampling_info(image):
    """Get the sampling info of an image for resample_to_ref"""
    if 'goudapath' in str(type(image)):
        image = read_meta(image)
    image_type = get_image_type(image)
    if image_type in ['sitk', 'sitkreader', 'smartimage']:
        result = {
            'size': image.GetSize(),
            'origin': image.GetOrigin(),
            'spacing': image.GetSpacing(),
            'direction': image.GetDirection()
        }
        if image_type == 'smartimage':
            result['dtype'] = image.dtype
        else:
            result['dtype'] = image.GetPixelID()
    elif image_type == 'itk':
        result = {
            'size': image.GetLargestPossibleRegion().GetSize(),
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': itk.GetArrayFromMatrix(image.GetDirection()).flatten(),
            'dtype': itk.template(image)[1][0]
        }
    else:
        raise ValueError("Unknown image type: {}".format(type(image)))
    return result


def compare_physical(image1, image2, check_dtype=False):
    info1 = image1 if isinstance(image1, dict) else get_sampling_info(image1)
    info2 = image2 if isinstance(image2, dict) else get_sampling_info(image2)
    if info1.keys() != info2.keys():
        raise ValueError('Mismatched keys between images')
    for key in info1.keys():
        if not check_dtype and key == 'dtype':
            continue
        if isinstance(info1[key], np.ndarray):
            key_check = np.allclose(info1[key], info2[key])
        elif gouda.is_iter(info1[key]):
            key_check = tuple(info1[key]) == tuple(info2[key])
        else:
            key_check = info1[key] == info2[key]
        if not key_check:
            return False
    return True


def compare_dicom_images(dicom_path1, dicom_path2):
    """Compare two dicom images and print out any differences in metadata"""
    image1, reader1 = io.read_dicom_as_sitk(dicom_path1)
    image2, reader2 = io.read_dicom_as_sitk(dicom_path2)
    found_diff = False

    image_check = compare_physical(image1, image2)
    if not image_check:
        print('Mismatched physical images')
        found_diff = True

    for slice_idx in range(image1.GetSize()[2]):
        keys1 = set(reader1.GetMetaDataKeys(slice_idx))
        keys2 = set(reader2.GetMetaDataKeys(slice_idx))
        merged = keys1.union(keys2)
        if len(merged) != len(keys1) or len(merged) != len(keys2):
            print('Slice {:03d} has mismatched keys - image 1 = {} keys, image 2 = {} keys'.format(slice_idx, len(keys1), len(keys2)))
            found_diff = True
            diff1 = keys1.difference(keys2)
            if len(diff1) != 0:
                print('\tKeys in 1 but not in 2')
                for key in diff1:
                    print('\t{}'.format(key))
            diff2 = keys2.difference(keys1)
            if len(diff2) != 0:
                print('\tKeys in 2 but not in 1')
                for key in diff2:
                    print('\t{}'.format(key))
            print()
        first_line = True
        for key in merged:
            val1 = reader1.GetMetaData(slice_idx, key)
            val2 = reader2.GetMetaData(slice_idx, key)
            if val1 != val2:
                found_diff = True
                if first_line:
                    print('Slice {:03d} has mismatched key values'.format(slice_idx))
                    first_line = False
                print('Key: {:7s}'.format(key))
                print('Image1: {}'.format(val1))
                print('Image2: {}'.format(val2))
    if not found_diff:
        print('No differences found')


def get_scaled_sampling_info(src_image, scaling_factor=0.5):
    dimension = src_image.GetDimension()
    src_size = np.array(src_image.GetSize())
    src_spacing = np.array(src_image.GetSpacing())
    physical_size = np.multiply(src_size - 1, src_spacing, where=src_size * src_spacing > 0, out=np.zeros(dimension))

    dst_size = np.round(src_size * scaling_factor).astype(np.int)
    dst_spacing = physical_size / (dst_size - 1)
    return {'size': dst_size.tolist(),
            'origin': src_image.GetOrigin(),
            'spacing': dst_spacing.tolist(),
            'direction': src_image.GetDirection()}


def resample_rescale(src_image, scaling_factor=0.5, scaling_template=None, outside_val=0.0, interp=sitk.sitkLinear):
    """Simple rescaling by a constant factor in each dimension.

    Parameters
    ----------
    src_image : str or SimpleITK.Image
        The image to rescale
    scaling_factor : float
        The factor to scale by (the default is 0.5)
    scaling_template : dict or SimpleITK.Image or SimpleITK.ImageFileReader
        A template to resample the image to instead of using a constant factor
    interp : int
        The interpolation method to use (the default is SimpleITK.sitkLinear)
    """
    if scaling_template is None:
        scaling_template = get_scaled_sampling_info(src_image, scaling_factor)

    if isinstance(scaling_template, (sitk.Image, sitk.ImageFileReader)):
        dst_size = scaling_template.GetSize()
    elif isinstance(scaling_template, dict):
        dst_size = scaling_template['size']
    else:
        raise ValueError("What is this?:{}".format(type(scaling_template)))
    dst_image = sitk.Image(dst_size, src_image.GetPixelIDValue())
    dst_image = copy_meta_from_ref(dst_image, scaling_template)
    dimension = src_image.GetDimension()

    dst_center = np.array(dst_image.TransformContinuousIndexToPhysicalPoint(np.array(dst_size) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(src_image.GetDirection())
    origin_shift = [src_o - dst_o for src_o, dst_o in zip(src_image.GetOrigin(), dst_image.GetOrigin())]
    transform.SetTranslation(origin_shift)
    centered_transform = sitk.Transform(transform)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(src_image.GetSize()) / 2.0
    img_center = np.array(src_image.TransformContinuousIndexToPhysicalPoint(img_center))
    img_center = transform.GetInverse().TransformPoint(img_center)
    centering_transform.SetOffset(img_center - dst_center)
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    return sitk.Resample(src_image, dst_image, centered_transform, interp, outside_val)


def copy_meta_from_ref(image, ref):
    if isinstance(ref, sitk.Image):
        image.CopyInformation(ref)
    elif isinstance(ref, sitk.ImageFileReader):
        image.SetOrigin(ref.GetOrigin())
        image.SetSpacing(ref.GetSpacing())
        image.SetDirection(ref.GetDirection())
    elif isinstance(ref, dict):
        image.SetOrigin(ref['origin'])
        image.SetSpacing(ref['spacing'])
        image.SetDirection(ref['direction'])
    else:
        raise ValueError("Unknown reference type: '{}'".format(type(ref)))
    return image


def padded_resize(image, size, background_value=0):
    image_type = get_image_type(image)
    if image_type != 'sitk':
        image = as_image(image).sitk_image
    cur_x, cur_y, cur_z = image.GetSize()
    if len(size) == 2:
        size = [size[0], size[1], cur_z]
    lower_x = (size[0] - cur_x) // 2
    upper_x = (size[0] - cur_x) - lower_x
    lower_y = (size[1] - cur_y) // 2
    upper_y = (size[1] - cur_y) - lower_y
    lower_z = (size[2] - cur_z) // 2
    upper_z = (size[2] - cur_z) - lower_z

    # if background_value is None:
    #     background_value, _ = get_image_range(image)

    result = sitk.ConstantPad(image,
                              [lower_x, lower_y, lower_z],
                              [upper_x, upper_y, upper_z],
                              background_value)
    if image_type == 'smartimage':
        return SmartImage(result)


def resample_to_unit_spacing(image, interp=sitk.sitkBSpline, outside_val=-1000, verbose=False):
    """Resample the input image to 1mm spacing"""
    input_size = np.array(image.GetSize())
    input_spacing = np.array(image.GetSpacing())
    input_origin = image.GetOrigin()
    input_direction = image.GetDirection()

    output_size = (input_spacing * input_size)
    output_spacing = np.array([1.0, 1.0, 1.0])
    if verbose:
        print(input_size, input_spacing)
        print(output_size, output_spacing)
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interp)
    resample_filter.SetOutputDirection(input_direction)
    resample_filter.SetOutputOrigin(input_origin)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetSize([int(s) for s in output_size])
    resample_filter.SetDefaultPixelValue(outside_val)
    resampled_image = resample_filter.Execute(image)

    return resampled_image


def resample_dryrun(size, spacing, new_spacing=None, new_size=None):
    """Given a reference size/spacing and a desired size/spacing, return the new physical parameters"""
    size = np.array(size)
    spacing = np.array(spacing)
    if new_spacing is not None:
        if not hasattr(new_spacing, '__len__') or len(new_spacing) == 1:
            new_spacing = [new_spacing, new_spacing, new_spacing]
        new_spacing = np.array(new_spacing)
        new_size = (size * spacing) / new_spacing
        return new_size, new_spacing
    elif new_size is not None:
        if len(new_size) == 2:
            new_size = list(new_size) + [size[-1]]
        new_size = np.array(new_size)
        new_spacing = (spacing * size) / new_size
        return new_size, new_spacing
    else:
        raise ValueError('Either new_spacing or new_size must not be None')


def read_meta(image_path, load_private_tags=True):
    if 'goudapath' in str(type(image_path)):
        image_path = image_path.path
    if os.path.isdir(image_path):
        dicom_dir = io.search_for_dicom(image_path)
        if len(dicom_dir) == 0:
            raise ValueError('No dicom files found at: {}'.format(image_path))
        elif len(dicom_dir) > 1:
            warnings.warn('Multiple dicom directories found - using first found: {}'.format(dicom_dir[0]))
        image_path = dicom_dir[0]
        dicom_files = sorted(glob.glob(os.path.join(image_path, '*.dcm')))
        image_path = dicom_files[0]
        print(image_path)
        if len(dicom_files) > 1:
            warnings.warn('Meta is only read for a single dcm file in the directory. Use GoudaMI.io.read_dicom_as_sitk for all slices.')

    file_reader = sitk.ImageFileReader()
    file_reader.SetLoadPrivateTags(load_private_tags)
    file_reader.SetFileName(image_path)
    file_reader.ReadImageInformation()
    return file_reader


def get_largest_object(image: ImageType, n_objects: int = 1) -> ImageType:
    """Remove all but the largest n objects from a label

    Parameters
    ----------
    image : ImageType
        The binary label image to evaluate
    n_objects : int, optional
        The number of objects to preserve, by default 1
    """
    image_type = get_image_type(image)
    if image_type != 'sitk':
        image = as_image(image).sitk_image
    components = sitk.ConnectedComponent(image)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.SetComputePerimeter(False)
    lfilter.Execute(components)
    label_sizes = [[lfilter.GetNumberOfPixels(label), label] for label in lfilter.GetLabels()]
    label_sizes = sorted(label_sizes, key=lambda x: x[0], reverse=True)
    changes = {}
    for idx, (_, label) in enumerate(label_sizes):
        changes[label] = 1 if idx < n_objects else 0
    output = sitk.ChangeLabel(components, changeMap=changes)
    output = sitk.Cast(output, image.GetPixelID())
    if image_type == 'smartimage':
        output = SmartImage(output)
    return output


def otsu_threshold(values, bins='auto'):
    """Get the value to use for a single point otsu threshold given an image"""
    values = as_view(values)
    hist, bin_edges = np.histogram(values.ravel(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


def compare_images(image1, image2):
    diff = sitk.Subtract(image1, image2)
    diff = sitk.Abs(diff)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(diff)
    print("Total Difference: {}".format(stats.GetSum()))
    print("Max Difference: {}".format(stats.GetMaximum()))


def get_bounds(label: ImageType, bg_val: float = 0) -> List[Tuple[int, int]]:
    """Get the corners of the bounding box/cube for the given binary label

    Returns
    -------
    List[Tuple[int, int]]
        A list of the (start, stop) indices for each axis
    """
    image_type = get_image_type(label)
    if image_type == 'numpy':
        if bg_val != 0:
            label = label != bg_val
        bounds = []
        for i in range(label.ndim):
            axis_check = np.any(label, axis=tuple([j for j in range(label.ndim) if j != i]))
            axis_range = np.where(axis_check == True) # noqa
            bounds.append([axis_range[0][0], axis_range[0][-1] + 1])
        return bounds
    if image_type == 'sitk':
        pass
    else:
        label = as_image(label).sitk_image
    if 'integer' not in label.GetPixelIDTypeAsString():
        label = sitk.Cast(label != bg_val, sitk.sitkUInt8)
    ndim = label.GetDimension()
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.SetComputePerimeter(False)
    filt.Execute(label)
    bounds = {}
    for label_idx in filt.GetLabels():
        label_bounds = filt.GetBoundingBox(label_idx)
        bounds[label_idx] = [[label_bounds[i], label_bounds[i] + label_bounds[i + ndim]] for i in range(ndim)]
    return bounds


def get_shared_bounds(*masks, target_label: int = 1, extent='max', as_slice: bool = False, padding: Union[int, Sequence[int], Sequence[tuple[int, int]]] = 0):
    """Return the minimum bounds that encompass all target masks

    Parameters
    ----------
    target_label : int, optional
        Target label index to find the bounds of, by default 1
    extent : str, optional
        Whether to return the shared bounds encompassing the label union ("max") or intersection ("min"), by default 'max'
    as_slice : bool, optional
        Whether to return the bounds as a slice, by default False
    padding : Union[int, Sequence[int], Sequence[tuple[int, int]]], optional
        The number of pixels to pad the bounding box by along each axis, by default 0

    Note
    ----
    padding can be a single int, an int for each axis, or a lower and upper padding for each axis

    Returns
    -------
    list | tuple
        either a list of [start, stop] for each axis or a tuple of slices
    """
    bounds = None
    for mask in masks:
        mask_bounds = get_bounds(mask)[target_label]
        if bounds is None:
            bounds = mask_bounds
            extent = gouda.force_len(extent, len(bounds))
        else:
            for idx in range(len(bounds)):
                if extent[idx] == 'max':
                    bounds[idx] = [min(bounds[idx][0], mask_bounds[idx][0]),
                                   max(bounds[idx][1], mask_bounds[idx][1])]
                elif extent[idx] == 'min':
                    bounds[idx] = [max(bounds[idx][0], mask_bounds[idx][0]),
                                   min(bounds[idx][1], mask_bounds[idx][1])]
    return pad_bounds(bounds, padding, as_slice=as_slice)


def pad_bounds(bounds, padding, image_size=None, as_slice=False):
    """Add padding to bounds

    Parameters
    ----------
    bounds : Iterable[Iterable[int, int]]
        The bounds to pad - should be in format [[lower_bound, upper_bound], ...] for each dimension
    padding : int | Iterable[int] | Iterable[Iterable[int, int]]
        The padding to apply, can be a single value for all padding, a single value per dimension, or a lower and upper value per dimension
    image_size : Optional[int | Iterable[int]]
        The maximum size to allow padding to extend to, by default None (no limit)
    as_slice : bool, optional
        Whether to return the bounds as a slice, by default False

    NOTE
    ----
    Padding will be applied to the lower bound and upper bound of each dimension. The lower bounds will not pad beyond 0 (4 padding to a start of 2 will be 0, not -2). If image_size is provided, the upper bound will not pad beyond the image size (4 padding to a stop of 98 with image size of 100 will be 100, not 102).
    """
    bounds = [item.copy() for item in bounds]
    padding = gouda.force_len(padding, len(bounds))
    if image_size is not None:
        image_size = gouda.force_len(image_size, len(bounds))
    for idx in range(len(bounds)):
        axis_pad = gouda.force_len(padding[idx], 2)
        bounds[idx][0] = max(bounds[idx][0] - axis_pad[0], 0)
        if image_size is None:
            bounds[idx][1] += axis_pad[1]
        else:
            bounds[idx][1] = min(bounds[idx][1] + axis_pad[1], image_size[idx])
    if as_slice:
        return tuple([slice(*b) for b in bounds])
    else:
        return bounds


def split_itk_image_channels(image):
    image_arr = itk.GetArrayViewFromImage(image)
    split_images = np.split(image_arr, image.GetNumberOfComponentsPerPixel(), axis=image.GetImageDimension())
    for i, sub_image in enumerate(split_images):
        sub_image = itk.GetImageFromArray(np.squeeze(sub_image))
        sub_image.SetOrigin(image.GetOrigin())
        sub_image.SetSpacing(image.GetSpacing())
        sub_image.SetDirection(image.GetDirection())
        split_images[i] = sub_image
    return split_images


def itk_cast_like(image, ref_image):
    """Cast an itk.Image to the same pixel type as another itk.Image

    Parameters
    ----------
    image : itk.Image
        The image to cast
    ref_image : itk.Image
        The image to reference for output pixel type
    """
    caster = itk.CastImageFilter[type(image), type(ref_image)].New()
    caster.SetInput(image)
    caster.Update()
    return caster.GetOutput()


def resample_group(images, ref_idx=0, ref_image=None, outside_val=-1000, interp=sitk.sitkBSpline):
    """Resample all images to the same physical space

    Parameters
    ----------
    images : list
        A list of SimpleITK.Image to merge
    ref_idx : int
        The index of the image from images to use as a reference
    ref_image : dict or SimpleITK.Image or path
        The image to use as reference - will override the ref_idx
    outside_val : int
        The value for background pixels
    interp : int
        The interpolator to use for resampling
    """
    # TODO - allow passing reference dict
    if ref_image is None:
        ref_image = images[ref_idx]
    ref = get_sampling_info(ref_image)
    for idx in range(len(images)):
        if idx == ref_idx:
            continue
        if compare_physical(images[idx], ref):
            continue
        images[idx] = resample_to_ref(images[idx], ref_image, outside_val=outside_val, interp=interp)
    return images


def sitk_zeros_like(image, dtype=None):
    if not isinstance(image, dict):
        image = get_sampling_info(image)
    if dtype is not None:
        image['dtype'] = dtype
    zero_image = sitk.Image(tuple(image['size'].tolist()), SmartType.as_sitk(image['dtype']))
    zero_image.SetOrigin(image['origin'])
    zero_image.SetSpacing(image['spacing'])
    zero_image.SetDirection(image['direction'])
    return zero_image


def get_objects_within_range(binary_image, min_size=100, max_size=np.inf, merge_objects=False):
    """Return all objects larger than min_size and smaller than max_size

    Parameters
    ----------
    binary_image : sitk.Image | SmartImage
        The binary image to draw objects from
    min_size : int
        The minimum number of pixels in objects to be included
    max_size : int
        The maximum number of pixels in objects to be included
    merge_objects : bool
        If False, each object gets a separate label in the returned image. Otherwise, they are all 1
    """
    if not isinstance(binary_image, (sitk.Image, SmartImage)):
        raise ValueError('requires a SimpleITK.Image object')
    labels = sitk.ConnectedComponent(binary_image)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.SetComputePerimeter(False)
    lfilter.Execute(labels)
    relabel_map = {}
    idx = 1
    for label in lfilter.GetLabels():
        label_area = lfilter.GetNumberOfPixels(label)
        if label_area < min_size or label_area > max_size:
            relabel_map[label] = 0
        elif merge_objects:
            relabel_map[label] = 1
        else:
            relabel_map[label] = idx
            idx += 1
    labels = sitk.ChangeLabel(labels, changeMap=relabel_map)
    return labels


def remove_small_items(label_img, min_size=20, use_physical_size=False):
    image_type = get_image_type(label_img)
    if image_type == 'smartimage':
        label_img = label_img.sitk_image
    components = sitk.ConnectedComponent(label_img)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.SetComputePerimeter(False)
    lfilter.Execute(components)
    labels = lfilter.GetLabels()
    changes = {}
    for label in labels:
        if use_physical_size:
            changes[label] = 0 if lfilter.GetPhysicalSize(label) < min_size else 1
        else:
            changes[label] = 0 if lfilter.GetNumberOfPixels(label) < min_size else 1
    component_mask = sitk.ChangeLabel(components, changeMap=changes)
    result = label_img * sitk.Cast(component_mask, label_img.GetPixelID())
    if image_type == 'smartimage':
        return SmartImage(result)
    else:
        return result


def reassign_small_objects(label, max_size=50, n_largest=0, neighbor_kernel=sitk.sitkCross, ignore_labels=None):
    """Assign small objects to the label of the object they share the most border with

    Parameters
    ----------
    label : ImageType
        label to correct
    max_size : int, optional
        Maximum size of objects to reassign (larger than this are not corrected), by default 50
    n_largest : int, optional
        If > 0, the number of objects to keep for each label class - all other objects are reassigned or removed, the default is 0
    neighbor_kernel : int, optional
        The SimpleITK KernelEnum for the structuring element used to check object borders, the default is 3 (SimpleITK.sitkCross)
    ignore_labels : list[int] | None, optional
        Labels to leave unchanged (other objects can still be reassigned to this class), the default is None

    Note
    ----
    If n_largest <= 0, then all separate objects with a size less than max_size will be reassigned to the class of the object they share the largest border with.
    If n_largest > 0, then the n_largest objects with the greatest physical size for each class are unchanged, and all other objects are reassigned to the class of the unchanged objects that they share the largest border with.
    In both cases, if a small object does not share any border, then it will be removed.
    """
    label = as_image(label)
    new_label = zeros_like(label)
    n_largest = max(n_largest, 0)  # negatives could cause issues
    if ignore_labels is None:
        ignore_labels = []

    if n_largest > 0:
        early_stopping = True
        for label_idx in label.unique():
            local_label = label == label_idx
            if label_idx in ignore_labels:
                new_label += local_label * label_idx
                continue
            local_cc = local_label.connected_components()
            local_stats = local_cc.label_shape_stats()
            objects = [(object_idx, local_stats.GetPhysicalSize(object_idx)) for object_idx in local_stats.GetLabels()]
            if len(objects) > n_largest:
                early_stopping = False
            objects = sorted(objects, key=lambda x: x[1], reverse=True)
            remapper = {}
            for object_idx, size in objects[:n_largest]:
                remapper[object_idx] = label_idx
            for object_idx, size in objects[n_largest:]:
                remapper[object_idx] = 0
            new_label += local_cc.change_label(remapper)
        if early_stopping:
            # Return early if no objects need to be removed/changed
            return label

    for label_idx in label.unique():
        local_label = label == label_idx
        if label_idx in ignore_labels:
            if n_largest <= 0:
                new_label += local_label * label_idx
            continue
        local_cc = local_label.connected_components()
        local_stats = local_cc.label_shape_stats()
        objects = [(object_idx, local_stats.GetPhysicalSize(object_idx)) for object_idx in local_stats.GetLabels()]
        objects = sorted(objects, key=lambda x: x[1], reverse=n_largest > 0)
        remapper = {}
        for object_idx, size in objects[:n_largest]:
            remapper[object_idx] = 0
        for object_idx, size in objects[n_largest:]:
            if size > max_size and not n_largest > 0:
                remapper[object_idx] = label_idx
                continue
            local_object = local_cc == object_idx
            object_bounds = get_bounds(local_object)[1]
            object_bounds = pad_bounds(object_bounds, 2, as_slice=True)

            local_object = local_object[object_bounds]
            if n_largest > 0:
                local_region = new_label[object_bounds]
            else:
                local_region = label[object_bounds]
            neighborhood = quick_dilate(local_object, radius=1, kernel=neighbor_kernel) - local_object
            neighbors = local_region.as_view()[neighborhood.as_array().astype(bool)]
            idx, counts = np.unique(neighbors, return_counts=True)
            idx = idx[np.argsort(counts)][::-1]
            counts = counts[np.argsort(counts)][::-1]
            if idx[0] == 0 and len(idx) > 1:
                neighbor = idx[1]
            else:
                neighbor = idx[0]
            remapper[object_idx] = int(neighbor)
        if len(remapper) > 0:
            local_cc = local_cc.change_label(remapper)
            new_label = new_label + local_cc.astype(new_label.dtype)

    return new_label


def remove_distant_items(label, distance_thresh: float = 5, size_thresh: float = 0):
    """Remove objects in the label that are distant from the largest object

    Parameters
    ----------
    label : ImageType
        Label image to remove items from
    distance_thresh : int, optional
        Maximum allowed distance from the largest object (in physcial units), by default 5
    size_thresh : int, optional
        Minimum physical size before an object is considered a separate entity (if <=0, this is ignored), by default 0

    Note
    ----
    This removes objects present in the label that are too far from the largest object. This is determined by measuring the closest points between the largest object and each other separate object. If size_thresh is set to a non-zero value, then objects that are larger than that physical size are never removed regardless of distance.
    """
    label = as_image(label)
    label_cc = label.connected_components()
    label_stats = label_cc.label_shape_stats()

    if label_stats.GetNumberOfLabels() > 1:
        largest_idx = (0, -1)
        for label_idx in label_stats.GetLabels():
            size = label_stats.GetPhysicalSize(label_idx)
            if size > largest_idx[0]:
                largest_idx = (size, label_idx)

        largest_label = label_cc == largest_idx[1]
        dist_map = sitk.SignedMaurerDistanceMap(largest_label.sitk_image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True, backgroundValue=0)
        dist_arr = sitk.GetArrayViewFromImage(dist_map)
        remapper = {}
        for label_idx in label_stats.GetLabels():
            if label_idx == largest_idx[1]:
                remapper[label_idx] = 1
                continue
            elif size_thresh > 0 and label_stats.GetPhysicalSize(label_idx) > size_thresh:
                remapper[label_idx] = 1
                continue
            local_dist = dist_arr[(label_cc == label_idx).as_view().astype(bool)].min()
            remapper[label_idx] = int(local_dist < distance_thresh)
        label_cc = label_cc.change_label(remapper)
        return label_cc.astype(label.dtype)
    else:
        return label


def get_total_hull(arr):
    """Get a convex hull encompasing all foreground of a 2d label"""
    import cv2
    result = np.zeros_like(arr)
    if arr.sum() == 0:
        return result
    contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    full_hull = cv2.convexHull(np.concatenate(contours, axis=0), False)
    cv2.drawContours(result, [full_hull], -1, 1, -1)
    return result


def get_ndim_hull(image):
    """Approximate the n-dimensional convex hull of the image label
    """
    image = as_image(image)
    image_arr = image.as_view()
    result = np.zeros(image_arr.shape, dtype=np.uint8)
    for dim in range(image_arr.ndim):
        slicer = [slice(None) for _ in range(image_arr.ndim)]
        for idx in range(image_arr.shape[dim]):
            slicer[dim] = idx
            hull = get_total_hull(image_arr[tuple(slicer)])
            result[tuple(slicer)] += hull
    result = as_image(result > 0)
    result.CopyInformation(image)
    return result


@wrap_image_func('sitk')
def get_label_hull(label):
    """Get the convex hull of each slice of a label image"""
    arr = sitk.GetArrayViewFromImage(label)
    result = np.zeros_like(arr)
    for slice_idx in range(arr.shape[0]):
        result[slice_idx] = get_total_hull(arr[slice_idx])
    result_img = sitk.GetImageFromArray(result)
    result_img.CopyInformation(label)
    return result_img


@wrap_numpy2numpy
def argmax(image, axis=-1, return_type=np.uint8):
    return image.argmax(axis=axis).astype(return_type)


def get_unique_labels(image, bg_val=-1):
    # NOTE: Assumes there is at least some background with value 0
    image = SmartImage(image).sitk_image
    label_filt = sitk.LabelShapeStatisticsImageFilter()
    label_filt.SetBackgroundValue(bg_val)
    label_filt.SetComputePerimeter(False)
    label_filt.Execute(image)
    return np.array(label_filt.GetLabels())


def get_num_objects(label: ImageType, min_size: float = 0, bg_val: float = 0):
    label_type = get_image_type(label)
    if label_type == 'smartimage':
        label = label.sitk_image
    elif label_type == 'itk':
        label = as_image(label).sitk_image
    if bg_val == 0 and min_size == 0:
        filt = sitk.ConnectedComponentImageFilter()
        filt.Execute(label)
        return filt.GetObjectCount()
    cc = sitk.ConnectedComponent(label)
    label_filt = sitk.LabelShapeStatisticsImageFilter()
    label_filt.SetBackgroundValue(bg_val)
    label_filt.SetComputePerimeter(False)
    label_filt.Execute(cc)
    if min_size <= 0:
        return label_filt.GetNumberOfLabels()
    count = 0
    for label_idx in label_filt.GetLabels():
        if label_filt.GetPhysicalSize(label_idx) >= min_size:
            count += 1
    return count


def cast_to_smallest_dtype(image: Union[sitk.Image, SmartImage]) -> sitk.Image:
    """Convert an image to the smallest integer dtype based on min/max values

    NOTE
    ----
    This will always truncate floating point values
    """
    if isinstance(image, SmartImage):
        image = image.sitk_image
    if 'float' in image.GetPixelIDTypeAsString():
        warnings.warn('truncating image values to integers')
    filt = sitk.MinimumMaximumImageFilter()
    filt.Execute(image)
    minimum = filt.GetMinimum()
    maximum = filt.GetMaximum()
    to_check = ['int8', 'int16', 'int32', 'int64']
    if minimum >= 0:
        to_check = ['u' + item for item in to_check]
    for item in to_check:
        dtype_range = np.iinfo(item)
        if minimum >= dtype_range.min and maximum <= dtype_range.max:
            dtype = item
            break
    else:
        raise ValueError('Unable to find proper dtype - this should never happen')
    return sitk.Cast(image, SmartType.as_sitk(dtype))


def get_value_range(image: sitk.Image) -> Tuple[float, float]:
    minimax_filt = sitk.MinimumMaximumImageFilter()
    minimax_filt.Execute(image)
    minimum = minimax_filt.GetMinimum()
    maximum = minimax_filt.GetMaximum()
    return minimum, maximum


def percentile_rescale(image: ImageType, upper_percentile: float, lower_percentile: float = None, output_min: float = 0, output_max: float = 1) -> ImageType:
    """Rescale an image based on percentiles

    Parameters
    ----------
    image : ImageType
        The image to rescale
    upper_percentile : float
        The upper percentile
    lower_percentile : float, optional
        The lower percentile, by default uses 1 - upper_percentile
    output_min : float, optional
        The minimum output value, by default 0
    output_max : float, optional
        The maximum output value, by default 0
    """
    image_type = get_image_type(image)
    image = as_image(image)
    if lower_percentile is None:
        lower_percentile = 100 - upper_percentile
    lower, upper = np.percentile(image.as_view(), [lower_percentile, upper_percentile])
    result = image.window(min=lower, max=upper, output_min=output_min, output_max=output_max)
    if image_type == 'sitk':
        return result.sitk_image
    elif image_type == 'itk':
        return as_image(result).itk_image
    return result


def check_small_diff(img1: sitk.Image, img2: sitk.Image, threshold: float = 1e-4, spec_thresholds: Optional[dict] = None, verbose: bool = True) -> bool:
    """Check whether differences in image size, spacing, or origin pass a defined threshold

    Parameters
    ----------
    img1: sitk.Image
        First image
    img2: sitk.Image
        Second image
    threshold: float
        The default threshold for all metric comparisons (the default is 1e-4)
    spec_thresholds: dict | None
        An optional dict with metrics as keys and metric-specific thresholds as values (the default is None)
    verbose: bool
        Whether to give a summary of any differences found (the default is True)

    Returns
    -------
    all_check: bool
        Returns true if any differences passed the thresholds
    """
    if spec_thresholds is None:
        spec_thresholds = {}
    prop1 = get_sampling_info(img1)
    prop2 = get_sampling_info(img2)
    all_check: bool = False
    for metric in ['size', 'spacing', 'origin']:
        metric_threshold = spec_thresholds.get(metric, threshold)
        diff = np.array(prop1[metric]) - np.array(prop2[metric])
        check = np.abs(diff) > metric_threshold
        if np.any(check) and verbose:
            # print({} vs {} -- Diff {}'.format(metric, str(prop1[metric]), str(prop2[metric]), str(diff)))
            print('Large difference in {} found'.format(metric))
            print('\tImage 1: {}'.format(prop1[metric]))
            print('\tImage 2: {}'.format(prop2[metric]))
            print('\tDiff   : {}'.format(diff))
        all_check = all_check or np.any(check)
    return all_check


def histeresis_threshold(prob_map, lower_thresh, peak_thresh, min_peak_size=100):
    # TODO - clean up
    if isinstance(prob_map, SmartImage):
        prob_map = prob_map.sitk_image
    minimum, maximum = get_value_range(prob_map)
    if peak_thresh > maximum:
        raise ValueError('Peak threshold cannot be greater than maximum image value')
    if lower_thresh > maximum:
        raise ValueError('Lower threshold cannot be greater than maximum image value')
    peak_map = sitk.BinaryThreshold(prob_map, peak_thresh, maximum)
    lower_map = sitk.BinaryThreshold(prob_map, lower_thresh, maximum)
    lower_cc = sitk.ConnectedComponent(lower_map)
    lower_cc_peaks = lower_cc * sitk.Cast(peak_map, lower_cc.GetPixelID())
    label_filt = sitk.LabelShapeStatisticsImageFilter()
    label_filt.SetComputePerimeter(False)
    label_filt.Execute(lower_cc_peaks)
    remap = {}
    lower_labels = get_unique_labels(lower_cc, bg_val=0)
    peak_labels = label_filt.GetLabels()
    for label_idx in lower_labels:
        label_idx = int(label_idx)
        if (label_idx not in peak_labels) or (label_filt.GetPhysicalSize(label_idx) < min_peak_size):
            remap[label_idx] = 0
        else:
            remap[label_idx] = 1
    result = sitk.ChangeLabel(lower_cc, changeMap=remap)
    return sitk.Cast(result, sitk.sitkUInt8)


def check_label_on_border(image: Union[sitk.Image, SmartImage]):
    if isinstance(image, SmartImage):
        image = image.sitk_image

    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.SetBackgroundValue(0)
    filt.ComputePerimeterOn()
    filt.Execute(image)

    results = {}
    for label in filt.GetLabels():
        results[label] = filt.GetPerimeterOnBorder(label) > 0
    return results


def add_to_empty(base_image: ImageType, add_image: ImageType, bg_val: float = 0) -> ImageType:
    """Add values from one label only to background regions of a base label

    Parameters
    ----------
    base_image : ImageType
        The label to add values to
    add_image : ImageType
        The label containing new values
    bg_val : float, optional
        The background value, by default 0

    Returns
    -------
    ImageType
        The result image with new values added to the bg of the base image
    """
    # TODO - can this be done without the inversion?
    src_type = get_image_type(base_image)
    base_image = as_image(base_image)
    add_image = as_image(add_image)
    inverse = (base_image == bg_val)
    inverse.astype(base_image.dtype, in_place=True)
    result = base_image + (add_image * inverse)
    if src_type == 'itk':
        return result.itk_image
    elif src_type == 'sitk':
        return result.sitk_image
    else:
        return result


def compare_relabel(label: SmartImage, neighbor: SmartImage) -> Tuple[SmartImage, SmartImage, bool, int]:
    """Reassign pieces of label to neighbor if they are not the largest label object and are touching neighbor

    Parameters
    ----------
    label : SmartImage
        The label that should only have 1 object
    neighbor : SmartImage
        A potentially neighboring label to the first
    """
    full_size = label.sum()
    objects = sitk.ConnectedComponent(label.sitk_image)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.SetComputePerimeter(False)
    filt.Execute(objects)
    mapper = {}
    remaining_obj = filt.GetNumberOfLabels()
    changed_neighbor = False
    for idx in filt.GetLabels():
        if filt.GetNumberOfPixels(idx) >= full_size * 0.5:
            mapper[idx] = 1
            continue
        merged = neighbor + (objects == idx)
        merged_cc = SmartImage(sitk.ConnectedComponent(merged.sitk_image))
        if merged_cc.max() == 1:
            mapper[idx] = 2
            remaining_obj -= 1
            changed_neighbor = True
        else:
            mapper[idx] = 0
            remaining_obj -= 1
    remapped = sitk.Cast(sitk.ChangeLabel(objects, changeMap=mapper), sitk.sitkUInt8)
    result_label = SmartImage(remapped == 1).astype('uint8')
    neighbor = neighbor + (remapped == 2)
    return result_label, neighbor, changed_neighbor, remaining_obj


def repair_label(raw_label: SmartImage, skip_labels: dict, border_tol: float = 1) -> SmartImage:
    raw_label = as_image(raw_label)
    to_check_label = zeros_like(raw_label)  # may need to be uint16 if num objects > 255

    to_check_lookup = {}
    max_idx = 0
    for label_idx in raw_label.unique()[1:]:
        if label_idx in skip_labels:
            continue
        fg = raw_label == label_idx
        cc_filt = sitk.ConnectedComponentImageFilter()
        cc = cc_filt.Execute(fg.sitk_image)
        num_objects = cc_filt.GetObjectCount()
        if num_objects < 255:
            cc = sitk.Cast(cc, sitk.sitkUInt8)
        else:
            raise ValueError('Too many objects for a single label')
        change_map = {}
        if num_objects != 1:
            largest = [0, -1]
            filt = sitk.LabelShapeStatisticsImageFilter()
            filt.SetComputePerimeter(False)
            filt.Execute(cc)
            for sublabel_idx in filt.GetLabels():
                change_map[sublabel_idx] = sublabel_idx + max_idx
                to_check_lookup[sublabel_idx + max_idx] = filt.GetBoundingBox(sublabel_idx)
                label_size = filt.GetNumberOfPixels(sublabel_idx)
                if label_size > largest[1]:
                    largest = [sublabel_idx, label_size]
            change_map[largest[0]] = 0
            to_check = sitk.ChangeLabel(cc, changeMap=change_map)
            raw_label = raw_label - (to_check != 0) * label_idx
            to_check_label = to_check_label + to_check
        max_idx += (num_objects - 1)

    change_map = {}
    for label_idx in to_check_lookup:
        x_start, y_start, z_start, x_size, y_size, z_size = to_check_lookup[label_idx]
        label_slice = (slice(x_start, x_start + x_size), slice(y_start, y_start + y_size), slice(z_start, z_start + z_size))
        to_check_small = to_check_label[label_slice] == label_idx

        check_dist = sitk.SignedMaurerDistanceMap(to_check_small.sitk_image, squaredDistance=False, useImageSpacing=True)
        check_dist_arr = sitk.GetArrayViewFromImage(check_dist)
        fixed_small = raw_label[label_slice]
        maybe_found = []
        for sublabel_idx in fixed_small.unique()[1:]:
            if sublabel_idx in skip_labels:
                continue
            contour = sitk.LabelContour((fixed_small == sublabel_idx).sitk_image, fullyConnected=True)
            contour_arr = sitk.GetArrayViewFromImage(contour).astype(np.bool_)
            dists = np.abs(check_dist_arr[contour_arr])
            border_val = (dists < border_tol).sum()
            if border_val > 0:
                maybe_found.append([sublabel_idx, border_val])
        maybe_found = sorted(maybe_found, key=lambda x: x[1], reverse=True)
        if len(maybe_found) > 0:
            change_map[label_idx] = maybe_found[0][0]
            print('Adding to {}'.format(maybe_found[0][0]), maybe_found)
        else:
            change_map[label_idx] = 0

    raw_label = raw_label + sitk.ChangeLabel(to_check_label.sitk_image, changeMap=change_map)
    return raw_label


def multiply_vector_image(image: sitk.Image, scalar: Union[float, Sequence[float]]) -> sitk.Image:
    """Multiply a vector image by a scalar

    Parameters
    ----------
    image : sitk.Image
        vector image to multiply
    scalar : Union[float, Sequence[float]]
        The scalar or list of scalars to multiply by
    """
    if not gouda.is_iter(scalar):
        scalar = (scalar,) * image.GetNumberOfComponentsPerPixel()
    assert len(scalar) == image.GetNumberOfComponentsPerPixel(), "Scalar must be a single value or a sequence of values the same length as the number of components in the image"
    return sitk.Compose([sitk.VectorIndexSelectionCast(image, i) * scalar[i % len(scalar)] for i in range(image.GetNumberOfComponentsPerPixel())])


@wrap_image_func('sitk')
def elastic_deformation(image: ImageType, sigma: Union[float, tuple[float, ...]] = 1.0, alpha: Union[float, tuple[float, ...]] = 2.0, interp=sitk.sitkLinear, seed: Optional[Union[np.random.Generator, int]] = None) -> sitk.Image:
    """Perform a random elastic deformation on the image.

    Parameters
    ----------
    image : ImageType
        Image to deform
    sigma : Union[float, tuple[float, ...]], optional
        The smoothness of the deformation - higher is smoother, by default 1.0
    alpha : Union[float, tuple[float, ...]], optional
        The magnitude of the deformation, by default 2.0
    seed : Optional[Union[np.random.Generator, int]], optional
        The seed or generator for random values, by default None

    NOTE
    ----
    sigma and alpha can either be single float values or tuples of values for each dimension
    """
    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)
    deformation = random.random(size=(*image.GetSize()[::-1], image.GetDimension())) * 2 - 1
    def_image = sitk.GetImageFromArray(deformation, isVector=True)
    def_image.CopyInformation(image)
    def_image = sitk.SmoothingRecursiveGaussian(def_image, sigma=sigma)
    def_image = multiply_vector_image(def_image, alpha)

    warp_filt = sitk.WarpImageFilter()
    warp_filt.SetInterpolator(interp)
    warp_filt.SetOutputParameteresFromImage(image)
    return warp_filt.Execute(image, def_image)


def quick_euler_3d(self,
                   rotation: Union[Sequence[float], npt.NDArray[np.floating]] = (0., ),
                   translation: Union[Sequence[int], npt.NDArray[np.integer]] = (0., ),
                   center: Optional[Sequence[float]] = None,
                   as_degrees: bool = False,) -> sitk.Euler3DTransform:
    """Short-cut method for getting a sitk.Euler3DTransform object

    Parameters
    ----------
    rotation : Union[Sequence[float], npt.NDArray[np.floating]], optional
        The rotation to apply along each axis, by default (0., )
    translation : Union[Sequence[int], npt.NDArray[np.integer]], optional
        The translation to apply along each axis, by default (0., )
    center : Optional[Sequence[float]], optional
        The center of the rotations, if None uses the image center, by default None
    as_degrees : bool, optional
        If True, assumes rotation is given in degrees, by default False

    Returns
    -------
    sitk.Euler3DTransform
        The resulting transform

    """
    (rotation, translation), count = gouda.match_len(rotation, translation, 3)

    rotation = np.array(rotation)
    if as_degrees:
        rotation = np.deg2rad(rotation)
    translation = np.array(translation)

    if center is None:
        center = self.GetCenter()
    if self.ndim == 3:
        transform = sitk.Euler3DTransform()
    else:
        raise NotImplementedError('SmartImage.euler_transform has only been added for 3D so far')

    transform.SetCenter(center)
    transform.SetRotation(*rotation.tolist())
    transform.SetTranslation(translation.tolist())
    return transform


@wrap_image_func('sitk')
def pad_to_same(*images: sitk.Image, bg_val: int = 0, upper_pad_only: bool = True, share_info: bool = False) -> list[sitk.Image]:
    """Pad all images to the same size.

    Parameters
    ----------
    *images : sitk.Image
        The images to pad
    bg_val : int, optional
        Background pad value, by default 0
    upper_pad_only : bool, optional
        Whether to only pad away from the origin
    share_info : bool, optional
        Whether to copy information from the first image across all images, by default False

    NOTE
    ----
    Only use share_info if you don't care about origin/spacing/direction of the images. Not all images get the same padding, so their origin will likely shift differently.
    """
    sizes = [np.array(image.GetSize()) for image in images]
    largest = np.max(sizes, axis=0)
    results = []
    shared_ref = None
    for image in images:
        image_size = np.array(image.GetSize())
        pad = largest - image_size
        upper_pad = (pad // 2).astype(int)
        lower_pad = (pad - upper_pad).astype(int)
        if upper_pad.sum() + lower_pad.sum() > 0:
            image = sitk.ConstantPad(image, lower_pad.tolist(), upper_pad.tolist(), constant=bg_val)
        if share_info:
            if shared_ref is None:
                shared_ref = image
            else:
                image.CopyInformation(shared_ref)
        results.append(image)
    return results


@wrap_image_func('sitk')
def pad_to_cube(img: sitk.Image, bg_val=0) -> sitk.Image:
    """Constant pad image so all sides are the same length.

    Parameters
    ----------
    img : sitk.Image
        Image to pad
    bg_val : int, optional
        The value to pad with, by default 0
    """
    size = img.GetSize()
    pad = [max(size) - s for s in size]
    upper_pad = [p // 2 for p in pad]
    lower_pad = [pad[i] - upper_pad[i] for i in range(len(pad))]
    return sitk.ConstantPad(img, lower_pad, upper_pad, constant=bg_val)


def label2vec(image: ImageType, bg_val=0, dtype='vector float32') -> SmartImage:
    """Divide an image of integers into vector components based on value

    Parameters
    ----------
    image : ImageType
        label image to convert into a vector
    bg_val : int, optional
        value of background pixels, by default 0
    dtype : str, optional
        Output type of the vector image, by default 'vector float32'


    NOTE
    ----
    This is most useful for interpolating a label image with multiple label classes, but you will get best results by using floating type vectors.
    """
    if not isinstance(image, sitk.Image):
        image = as_image(image).sitk_image
    labels = np.unique(sitk.GetArrayViewFromImage(image))
    result = sitk.Compose([image == label for label in labels if label != bg_val])
    result = as_image(result).astype(dtype)
    return result


def vec2label(vec: ImageType) -> SmartImage:
    """Combine a vector of binary images into a label image"""
    if not isinstance(vec, sitk.Image):
        vec = as_image(vec).sitk_image
    result = None
    for idx in range(vec.GetNumberOfComponentsPerPixel()):
        image = sitk.VectorIndexSelectionCast(vec, idx) * (idx + 1)
        image = (image > 0.5) * (idx + 1)
        result = image if result is None else sitk.Maximum(result, image)
    return as_image(result).astype('uint8')


def total_variation(image: ImageType):
    """Compute the anisotropic total variation of the image

    Equation source: https://www.wikiwand.com/en/Total_variation_denoising
    """
    image = as_image(image).as_view()
    tv = 0
    for axis in range(image.ndim):
        slicer_front = [slice(None)] * image.ndim
        slicer_front[axis] = slice(1, None)
        slicer_front = tuple(slicer_front)
        slicer_back = [slice(None)] * image.ndim
        slicer_back[axis] = slice(None, -1)
        slicer_back = tuple(slicer_back)
        tv += np.abs(image[slicer_front] - image[slicer_back]).sum()
    return tv


def merge_close_objects(label: ImageType, tol: int = 3):
    """Merge objects that are within each other's bounding boxes

    Parameters
    ----------
    label : ImageType
        The label image to merge - separate objects should have separate values
    tol : int, optional
        The number of voxels to expand the bounding box by in each direction when finding overlap, by default 3

    Returns
    -------
    ImageType
        The merged label image
    """
    bounds = get_bounds(label)
    changed = False
    result_map = {idx: idx for idx in bounds.keys()}
    to_check = sorted(list(bounds.keys()))
    for key in to_check:
        key_bounds = tuple([slice(b[0] - tol, b[1] + tol) for b in bounds[key]])
        merge_keys = label[key_bounds].unique()
        if len(merge_keys) > 1:
            changed = True
            min_key = min([result_map[subkey] for subkey in merge_keys])
            for subkey in merge_keys:
                result_map[subkey] = min_key
        unique_vals = set([result_map[subkey] for subkey in result_map.keys()])
        if len(unique_vals) == 1:
            break
    if changed:
        return label.change_label(result_map)
    else:
        return label


def get_overlapping_objects(label: ImageType, pred: ImageType, tol: int = 3, run_connected_components: bool = False):
    """Find overlapping object between two label images based on bounding boxes

    Parameters
    ----------
    label : ImageType
        The base image to find overlaps from
    pred : ImageType
        The image to find overlaps in
    tol : int, optional
        The number of pixels to expand the bounding box by in each direction when finding overlap, by default 3
    run_connected_components : bool, optional
        Whether to run connected components on the two label images before comparing, by default False

    Note
    ----
    If run_connected_components is False, it assumes that all objects have a unique value within their respective images.

    Returns
    -------
    dict
        The keys are the label values in the label image, and the values are the label values in the pred image that overlap with the key
    """
    label = as_image(label)
    pred = as_image(pred)
    if run_connected_components:
        label = label.cc
        pred = pred.cc
    label = merge_close_objects(label, tol=tol)
    pred = merge_close_objects(pred, tol=tol)

    label_bounds = get_bounds(label)
    overlaps = {}
    for key in label.unique():
        key_bounds = tuple([slice(b[0] - tol, b[1] + tol) for b in label_bounds[key]])
        overlap_keys = pred[key_bounds].unique()
        overlaps[key] = overlap_keys
    return overlaps


@wrap_image_func('smart')
def get_vessel_image(image, hessian_sigma=1.5, alpha1=0.5, alpha2=2.0, clip_input=('min', -500), clip_output=(15, 'max'), output_range=None):
    """Get the vesselness measure of an input image (intended for pulmonary CT)

    Parameters
    ----------
    image : GoudaMI.SmartImage
        Input image
    hessian_sigma : float, optional
        Sigma value for the hessian recursive gaussian filter, by default 1.0
    alpha1 : float, optional
        Vesselness parameter, by default 0.5
    alpha2 : float, optional
        Vesselness parameter, by default 2.0
    output_range : Optional[tuple[float, float]], optional
        The range to scale the output to, by default None

    Note
    ----
    Adapted from: https://examples.itk.org/src/filtering/imagefeature/segmentbloodvessels/documentation
    Algorithm from: http://www.image.med.osaka-u.ac.jp/member/yoshi/paper/linefilter.pdf
    """
    if clip_input is not None:
        input_min, input_max = clip_input
        if input_min == 'min':
            input_min = image.min()
        if input_max == 'max':
            input_max = image.max()
        image = image.window(min=input_min, max=input_max)
    hessian_image = itk.hessian_recursive_gaussian_image_filter(image.itk_image, sigma=hessian_sigma)
    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.F].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)
    vesselness_filter.Update()
    vessel_image = SmartImage(vesselness_filter.GetOutput())
    if output_range is not None:
        result_min, result_max = output_range
        if result_min == 'min':
            result_min = vessel_image.min()
        if result_max == 'max':
            result_max = vessel_image.max()
    if clip_output is not None:
        output_min, output_max = clip_output
        if output_min == 'min':
            output_min = vessel_image.min()
        if output_max == 'max':
            output_max = vessel_image.max()
        if output_range is not None:
            vessel_image = vessel_image.window(min=output_min, max=output_max, output_min=result_min, output_max=result_max)
        else:
            vessel_image = vessel_image.window(min=output_min, max=output_max)
    elif output_range is not None:
        vessel_image = vessel_image.window(output_min=result_min, output_max=result_max)
    return vessel_image

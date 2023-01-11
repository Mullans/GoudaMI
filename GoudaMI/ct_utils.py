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
from GoudaMI.constants import MIN_INTENSITY_PULMONARY_CT, SmartType
from GoudaMI.convert import as_view, wrap_numpy2numpy, wrap_sitk
from GoudaMI.optional_imports import itk
from GoudaMI.smart_image import (ImageRefType, ImageType, SmartImage, as_image, as_image_type, get_image_type, zeros_like)

# NOTE - reference for some interfacing: https://github.com/SimpleITK/SimpleITK/blob/4aabd77bddf508c1d55519fbf6002180a08f9208/Wrapping/Python/Python.i#L764-L794


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


def quick_open(img, radius=3):
    """Shortcut method for applying the sitk BinaryMorphologicalOpeningImageFilter"""
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    opener = sitk.BinaryMorphologicalOpeningImageFilter()
    opener.SetKernelType(sitk.sitkBall)
    opener.SetKernelRadius(radius)
    opener.SetForegroundValue(1)
    opener.SetBackgroundValue(0)
    return opener.Execute(img)


def quick_close(img, radius=3):
    """Shortcut method for applying the sitk BinaryMorphologicalClosingImageFilter"""
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    closer = sitk.BinaryMorphologicalClosingImageFilter()
    closer.SetKernelType(sitk.sitkBall)
    closer.SetKernelRadius(radius)
    closer.SetForegroundValue(1)
    return closer.Execute(img)


def quick_dilate(img, radius=3):
    """Shortcut method for applying the sitk BinaryDilateImageFilter"""
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    dil_filter = sitk.BinaryDilateImageFilter()
    dil_filter.SetKernelType(sitk.sitkBall)
    dil_filter.SetKernelRadius(radius)
    dil_filter.SetForegroundValue(1)
    dil_filter.SetBackgroundValue(0)
    return dil_filter.Execute(img)


def quick_erode(img, radius=3):
    """Shortcut for applying the sitk BinaryErodeImageFilter"""
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    dil_filter = sitk.BinaryErodeImageFilter()
    dil_filter.SetKernelType(sitk.sitkBall)
    dil_filter.SetKernelRadius(radius)
    dil_filter.SetForegroundValue(1)
    dil_filter.SetBackgroundValue(0)
    return dil_filter.Execute(img)


def faster_mask_body(image, resample=True):
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
        mask = faster_mask_body(image)
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


def compare_physical(image1, image2):
    info1 = image1 if isinstance(image1, dict) else get_sampling_info(image1)
    info2 = image2 if isinstance(image2, dict) else get_sampling_info(image2)
    if info1.keys() != info2.keys():
        raise ValueError('Mismatched keys between images')
    for key in info1.keys():
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


def read_meta(image_path):
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
    if label.GetPixelID() != sitk.sitkUInt8:
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


def get_shared_bounds(mask1, mask2):
    bounds1, bounds2 = get_bounds(mask1)[1], get_bounds(mask2)[1]
    shared_bounds = [[max(bounds1[i], bounds2[i]), min(bounds1[i], bounds2[i]), None] for i in range(len(bounds1))]
    return shared_bounds


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
    ref = get_sampling_info(images[ref_idx])
    for idx in range(len(images)):
        if idx == ref_idx:
            continue
        if compare_physical(images[idx], ref):
            continue
        images[idx] = resample_to_ref(images[idx], images[ref_idx], outside_val=outside_val, interp=interp)
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


def remove_small_items(label_img, min_size=20):
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
        changes[label] = 0 if lfilter.GetNumberOfPixels(label) < min_size else 1
    component_mask = sitk.ChangeLabel(components, changeMap=changes)
    result = label_img * sitk.Cast(component_mask, label_img.GetPixelID())
    if image_type == 'smartimage':
        return SmartImage(result)
    else:
        return result


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


def get_label_hull(label):
    """Get the convex hull of each slice of a label image"""
    if isinstance(label, SmartImage):
        label = label.sitk_image
    elif isinstance(label, sitk.Image):
        pass
    else:
        raise NotImplementedError('Not implemented for type `{}` yet'.format(type(label)))

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


@wrap_sitk
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


def wrap_bounds(func, bg_val=0):
    """Wraps a method so that it only operates on the non-zero bounds of an image and re-pads it afterwards."""
    @functools.wraps(func)
    def wrapped_func(image, *args, **kwargs):
        image_type = get_image_type(image)
        image = as_image(image)
        bounds = get_bounds(image, bg_val=bg_val)[1]
        bound_slice = [slice(item[0], item[1]) for item in bounds]
        result = func(image[bound_slice].sitk_image, *args, **kwargs)
        if not isinstance(result, (sitk.Image, SmartImage, np.ndarray)):
            return result

        low_padding = [int(item[0]) for item in bounds]
        upper_padding = [int(side - item[1]) for side, item in zip(image.GetSize(), bounds)]
        result = as_image(result)
        result = result.apply(sitk.ConstantPad, low_padding, upper_padding, bg_val)
        return as_image_type(result, image_type)
    return wrapped_func


@wrap_sitk
def pad_to_same(*images: sitk.Image, bg_val: int = 0, share_info: bool = False) -> list[sitk.Image]:
    """Pad all images to the same size.

    Parameters
    ----------
    *images : sitk.Image
        The images to pad
    bg_val : int, optional
        Background pad value, by default 0
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


@wrap_sitk
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


def label2vec(image: ImageType, bg_val=0) -> SmartImage:
    """Divide an image of integers into vector components based on value"""
    if not isinstance(image, sitk.Image):
        image = as_image(image).sitk_image
    labels = np.unique(sitk.GetArrayViewFromImage(image))
    result = sitk.Compose([image == label for label in labels if label != bg_val])
    return as_image(result)


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

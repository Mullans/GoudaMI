import glob
import os
import warnings
from collections.abc import Iterable

import gouda

try:
    import itk
except ImportError:
    pass  # moved warning to __init__.py
    # warnings.warn("Could not import ITK module - some methods may not work")
    
import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from . import io
from .constants import DTYPE_MATCH_ITK, DTYPE_MATCH_SITK
from .smart_image import SmartImage
from .convert import wrap4numpy

MAX_INTENSITY = 500
MIN_INTENSITY = -1000


def quick_save(data, label=None, skip_existing_filenames=False):
    """Quickly save an image/label in the notebooks' scratch folder

    Parameters
    ----------
    data: array_like
        The data to save
    label: str
        The full or extension-less filename to save the data under (the default is None)
    skip_existing_filenames: bool
        Whether to skip saving the file if the label is already taken (the default is False)
    """
    scratch_dir = gouda.GoudaPath(os.getcwd() + '/scratch')
    gouda.ensure_dir(scratch_dir)
    if label is None:
        label = 'default'
    elif label[0] == '.':
        label = 'default' + label

    if '.' not in label:
        if hasattr(data, '__array__'):
            label = label + '.npy'
            if skip_existing_filenames:
                if scratch_dir(label).exists():
                    return
            data = np.squeeze(np.array(data))
            np.save(gouda.next_filename(scratch_dir(label).abspath), data)
        elif isinstance(data, sitk.Image):
            label = label + '.nii'
            if skip_existing_filenames:
                if scratch_dir(label).exists():
                    return
            sitk.WriteImage(data, gouda.next_filename(scratch_dir(label).abspath))
        elif isinstance(data, (dict, list)):
            label = label + '.json'
            if skip_existing_filenames:
                if scratch_dir(label).exists():
                    return
            gouda.save_json(data, gouda.next_filename(scratch_dir(label).abspath))
        else:
            raise ValueError("Unidentified data type")
    else:
        if skip_existing_filenames:
            if scratch_dir(label).exists():
                return
        if hasattr(data, '__array__'):
            data = np.squeeze(np.array(data))
            if '.nii' in label or '.nrrd' in label:
                data = sitk.GetImageFromArray(data)
                sitk.WriteImage(data, gouda.next_filename(scratch_dir(label).abspath))
            else:
                np.save(gouda.next_filename(scratch_dir(label).abspath), data)
        elif isinstance(data, sitk.Image):
            sitk.WriteImage(data, gouda.next_filename(scratch_dir(label).abspath))
        elif isinstance(data, (dict, list)):
            gouda.save_json(data, gouda.next_filename(scratch_dir(label).abspath))
        else:
            raise ValueError("Unidentified data type")


def clip_image(image, low=MIN_INTENSITY, high=MAX_INTENSITY):
    """Clip the intensity values in the image to a given range.

    Parameters
    ----------
    image : SimpleITK.Image
        The image to clip
    low: int
        The minimum value to use (the default is -1000)
    high: int
        The maximum value to use (the default is 500)

    NOTE: UI-Lung values: [-1024, 1024], NVidia Lung Values [-1000, 500]"""
    if isinstance(image, SmartImage):
        image = image.sitk_image
    # image = sitk.Threshold(image, -32768, high, high)
    # image = sitk.Threshold(image, low, high, low)
    return sitk.IntensityWindowing(image, low, high, low, high)
    return image


def body_binary_thresh(img, min_val=-500, max_val=2000):
    """Threshold a CT to best show the body vs background.

    NOTE: These min/max values are for thresholding the body, not the lungs"""
    bin_filter = sitk.BinaryThresholdImageFilter()
    bin_filter.SetOutsideValue(0)
    bin_filter.SetInsideValue(1)
    bin_filter.SetLowerThreshold(min_val)
    bin_filter.SetUpperThreshold(max_val)
    bin_img = bin_filter.Execute(img)
    return bin_img


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


def fill2d(arr, force_slices=False):
    if isinstance(arr, sitk.Image) and not force_slices:
        arr = sitk.ConstantPad(arr, [0, 0, 1], [0, 0, 1], 1)
        arr = sitk.BinaryFillhole(arr)
        return arr[:, :, 1:-1]
    if isinstance(arr, (sitk.Image, SmartImage)):
        pass
    output = np.zeros_like(arr)
    for idx in range(arr.shape[0]):
        check_slice = arr[idx]
        # check_slice = scipy.ndimage.binary_dilation(check_slice)
        filled = scipy.ndimage.binary_fill_holes(check_slice)
        output[idx] = filled
    return output


def fill_slices(arr, dilate=0, erode=0, axis=0):
    import skimage.morphology
    src_img = None
    if isinstance(arr, (sitk.Image, SmartImage)):
        src_img = arr
        arr = sitk.GetArrayFromImage(arr)
    output = np.zeros_like(arr)
    slices = [slice(None)] * arr.ndim
    structs = [skimage.morphology.disk(dilate), skimage.morphology.disk(erode)]

    for idx in range(arr.shape[axis]):
        slices[axis] = slice(idx, idx + 1)
        arr_slice = np.squeeze(arr[tuple(slices)])
        if dilate > 0:
            arr_slice = skimage.morphology.binary_dilation(arr_slice, footprint=structs[0])
        arr_slice = scipy.ndimage.binary_fill_holes(arr_slice)
        if erode > 0:
            arr_slice = skimage.morphology.binary_erosion(arr_slice, footprint=structs[0])
        output[tuple(slices)] = arr_slice
    if src_img is not None:
        output = sitk.GetImageFromArray(output)
        output.CopyInformation(src_img)
    return output


def get_largest_n_objects(binary_image, n=1):
    if not isinstance(binary_image, sitk.Image):
        raise ValueError("mask_body requires a SimpleITK.Image object")
    labels = sitk.ConnectedComponent(binary_image)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(labels)
    body_label = [-1, -1]
    for label in lfilter.GetLabels():
        label_area = lfilter.GetNumberOfPixels(label)
        if label_area > body_label[1]:
            body_label = [label, label_area]
    bin_img = sitk.Equal(labels, body_label)
    return bin_img


def mask_body(image, opening_size=1):
    """Generate a mask of the body in a 3D CT"""
    if not isinstance(image, sitk.Image):
        raise ValueError("mask_body requires a SimpleITK.Image object")
    bin_img = sitk.RecursiveGaussian(image, 3)
    bin_img = sitk.BinaryThreshold(bin_img, -500, 10000, 1, 0)
    if opening_size > 0:
        bin_img = sitk.BinaryMorphologicalOpening(bin_img, [opening_size] * image.GetDimension(), sitk.sitkBall, 0, 1)
    labels = sitk.ConnectedComponent(bin_img)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(labels)
    body_label = [-1, -1]
    for label in lfilter.GetLabels():
        label_area = lfilter.GetNumberOfPixels(label)
        if label_area > body_label[1]:
            body_label = [label, label_area]
    bin_img = sitk.Equal(labels, body_label[0])
    bin_img = sitk.BinaryMorphologicalClosing(bin_img, [3] * image.GetDimension(), sitk.sitkBall, 1)
    filled_labels = fill2d(bin_img)
    return filled_labels


def faster_mask_body(image, resample=True):
    # Note - rounding error may cut off the top slice during resampling
    src_image = image
    if resample:
        image = resample_iso_by_slice_size(image, 128, interp=sitk.sitkLinear)
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(3)
    image = median_filter.Execute(image)
    # bin_img = sitk.Greater(image, -500)
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
        bin_img = resample_to_ref(bin_img, src_image, interp=sitk.sitkNearestNeighbor, outside_val=0)
    return bin_img


def remove_background(image, mask):
    return sitk.Mask(image, mask, outsideValue=MIN_INTENSITY, maskingValue=0)


def crop_image_to_mask(image, mask=None, crop_z=False, crop_quantile=50, return_bounds=False):
    if mask is None:
        mask = mask_body(image)
    elif isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)

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


def clean_segmentation(lung_image, lung_threshold=0.1, body_mask=None):
    import skimage.measure
    if isinstance(lung_image, sitk.Image):
        source_arr = sitk.GetArrayFromImage(lung_image)
    else:
        source_arr = np.copy(lung_image)
    if body_mask is not None:
        source_arr[~body_mask.astype(np.bool)] = 0
    if 'float' in str(source_arr.dtype):
        lung_arr = source_arr > lung_threshold
    else:
        lung_arr = source_arr

    labels = skimage.measure.label(lung_arr, connectivity=1, background=0)
    new_labels = np.zeros_like(lung_arr)
    bins = np.bincount(labels.flat)
    pos_lungs = (np.argwhere(bins[1:] > 100000) + 1).flatten()
    if len(pos_lungs) > 2:
        raise ValueError('Too Many Lungs')
    elif len(pos_lungs) == 1:
        warnings.warn('Only single segmented object detected')
        lungs = pos_lungs[0]
        new_labels[labels == lungs] = source_arr[labels == lungs]
    elif len(pos_lungs) == 2:
        lung1, lung2 = pos_lungs
        new_labels[labels == lung1] = source_arr[labels == lung1]
        new_labels[labels == lung2] = source_arr[labels == lung2]
    else:
        raise ValueError('No segmentated objects found')
    if isinstance(lung_image, sitk.Image):
        clean_image = sitk.GetImageFromArray(new_labels)
        clean_image.CopyInformation(lung_image)
        return clean_image
    return new_labels


def resample_iso_by_slice_size(image, output_size, outside_val=MIN_INTENSITY, interp=sitk.sitkBSpline, presmooth=False):
    """Resample an image to a given slice size and enforce equal spacing in all dimensions"""
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


def resample_to_ref(im, ref, outside_val=0, interp=sitk.sitkNearestNeighbor):
    """Resample an image to match a reference image"""
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetInterpolator(interp)
    resampleFilter.SetDefaultPixelValue(outside_val)
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
    resampleIm = resampleFilter.Execute(im)
    return resampleIm


def get_sampling_info(image):
    """Get the sampling info of an image for resample_to_ref"""
    if isinstance(image, (str, gouda.GoudaPath)):
        image = read_meta(image)
    if isinstance(image, (sitk.Image, sitk.ImageFileReader, SmartImage)):
        result = {
            'size': image.GetSize(),
            'origin': image.GetOrigin(),
            'spacing': image.GetSpacing(),
            'direction': image.GetDirection()
        }
        if isinstance(image, SmartImage):
            result['dtype'] = image.dtype
        else:
            result['dtype'] = image.GetPixelID()
    elif isinstance(image, itk.Image):
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
        if isinstance(info1[key], Iterable):
            key_check = tuple(info1[key]) == tuple(info2[key])
        else:
            key_check = info1[key] == info2[key]
        if not key_check:
            return False
    return True


def compare_dicom_images(dicom_path1, dicom_path2):
    """Compare two dicom images and print out any differences"""
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


def apply_scaling(src_image, scaling_factor=0.5, scaling_template=None, outside_val=0.0, interp=sitk.sitkLinear):
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

    return sitk.ConstantPad(image,
                            [lower_x, lower_y, lower_z],
                            [upper_x, upper_y, upper_z],
                            background_value)


def resample_to_isocube(image, output_size=300, interp=sitk.sitkBSpline, outside_val=MIN_INTENSITY):
    """Resize an image to have equal spacing and sizing in each axis, and pad
    with background value to preserve aspect ratio"""
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    input_origin = image.GetOrigin()
    input_direction = image.GetDirection()

    input_size = np.array(input_size)
    input_spacing = np.array(input_spacing)

    # resample isotropic
    mid_size = (input_spacing * input_size) / input_spacing.min()
    mid_spacing = input_spacing.min()

    # get output size/spacing
    output_spacing = (mid_spacing * mid_size.max()) / output_size
    mid_size = ((input_spacing * input_size) / output_spacing).astype(np.int)
    output_spacing = (input_spacing * input_size) / mid_size

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interp)
    resample_filter.SetOutputDirection(input_direction)
    resample_filter.SetOutputOrigin(input_origin)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetSize([int(s) for s in mid_size])
    resample_filter.SetDefaultPixelValue(outside_val)
    resampled_image = resample_filter.Execute(image)

    cur_x, cur_y, cur_z = mid_size
    lower_x = int((output_size - cur_x) // 2)
    upper_x = int((output_size - cur_x) - lower_x)
    lower_y = int((output_size - cur_y) // 2)
    upper_y = int((output_size - cur_y) - lower_y)
    lower_z = int((output_size - cur_z) // 2)
    upper_z = int((output_size - cur_z) - lower_z)
    lower_bounds = [lower_x, lower_y, lower_z]
    upper_bounds = [upper_x, upper_y, upper_z]
    resampled_image = sitk.ConstantPad(resampled_image, lower_bounds, upper_bounds, outside_val)
    return resampled_image


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


def whatif(size, spacing, new_spacing=None, new_size=None):
    size = np.array(size)
    spacing = np.array(spacing)
    if new_spacing is not None:
        if not hasattr(new_spacing, '__len__') or len(new_spacing) == 1:
            new_spacing = [new_spacing, new_spacing, new_spacing]
        new_spacing = np.array(new_spacing)
        new_size = (size * spacing) / new_spacing
        return new_size
    elif new_size is not None:
        if len(new_size) == 2:
            new_size = list(new_size) + [size[-1]]
        new_size = np.array(new_size)
        new_spacing = (spacing * size) / new_size
        return new_spacing
    else:
        raise ValueError('Either new_spacing or new_size must not be None')


def read_meta(image_path):
    if isinstance(image_path, gouda.GoudaPath):
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


def isocube_to_source(cube_path, cropped_path=None, original_path=None, label=None, outside_val=MIN_INTENSITY):
    if cropped_path is None:
        cropped_path = cube_path.replace('Isocubes', 'CroppedImages')
        cropped_path = cropped_path.replace('SampledCubes', 'Images')
    if original_path is None:
        original_path = cube_path.replace('Isocubes', 'Images')
        original_path = original_path.replace('SampledCubes', 'Images')
    cube_image = sitk.ReadImage(cube_path)
    crop_image = sitk.ReadImage(cropped_path)
    original_image = sitk.ReadImage(original_path)

    if label is None:
        to_restore = cube_image
        interp = sitk.sitkBSpline
    else:
        if isinstance(label, np.ndarray):
            to_restore = sitk.GetImageFromArray(label)
            to_restore.CopyInformation(cube_image)
        interp = sitk.sitkNearestNeighbor

    crop_size = crop_image.GetSize()
    crop_space = crop_image.GetSpacing()
    # crop_origin = crop_image.GetOrigin()
    # crop_direction = crop_image.GetDirection()

    cube_size = cube_image.GetSize()
    # cube_space = cube_image.GetSpacing()

    # Un-pad
    input_spacing = np.array(crop_space)
    input_size = np.array(crop_size)
    output_size = np.array(cube_size)[0]

    mid_size = (input_spacing * input_size) / input_spacing.min()
    mid_spacing = input_spacing.min()

    output_spacing = (mid_spacing * mid_size.max()) / 300
    mid_size = ((input_spacing * input_size) / output_spacing).astype(np.int)
    output_spacing = (input_spacing * input_size) / mid_size

    cur_x, cur_y, cur_z = mid_size
    lower_x = int((output_size - cur_x) // 2)
    upper_x = int((output_size - cur_x) - lower_x)
    lower_y = int((output_size - cur_y) // 2)
    upper_y = int((output_size - cur_y) - lower_y)
    lower_z = int((output_size - cur_z) // 2)
    upper_z = int((output_size - cur_z) - lower_z)
    lower_bounds = [lower_x, lower_y, lower_z]
    upper_bounds = [upper_x, upper_y, upper_z]

    xslice, yslice, zslice = [slice(lower_bounds[i], cube_size[i] - upper_bounds[i]) for i in range(3)]
    to_restore = to_restore[xslice, yslice, zslice]

    # Resample to crop space
    upsampled = resample_to_ref(to_restore, crop_image, interp=interp, outside_val=outside_val)

    # Un-Crop to original image
    upsampled_origin = original_image.TransformPhysicalPointToIndex(upsampled.GetOrigin())
    original_size = original_image.GetSize()
    upsampled_size = upsampled.GetSize()

    upper_bounds = [original_size[i] - upsampled_size[i] - upsampled_origin[i] for i in range(3)]
    restored = sitk.ConstantPad(upsampled, upsampled_origin, upper_bounds, outside_val)
    restored.CopyInformation(original_image)
    return restored


def get_surface(image, connectivity=1):
    image = image.astype(np.bool)
    conn = scipy.ndimage.morphology.generate_binary_structure(image.ndim, connectivity)
    return image ^ scipy.ndimage.morphology.binary_erosion(image, conn)


def get_distances(image_1, image_2, sampling=1, connectivity=1):
    surf_1 = get_surface(image_1, connectivity=connectivity)
    surf_2 = get_surface(image_2, connectivity=connectivity)
    dta = scipy.ndimage.morphology.distance_transform_edt(~surf_1, sampling)
    dtb = scipy.ndimage.morphology.distance_transform_edt(~surf_2, sampling)
    return np.concatenate([np.ravel(dta[surf_2 != 0]), np.ravel(dtb[surf_1 != 0])])


def get_biggest_object(mask):
    components = sitk.ConnectedComponent(mask)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(components)
    labels = lfilter.GetLabels()
    biggest = [-1, -1]
    for label in labels:
        value = lfilter.GetNumberOfPixels(label)
        if value > biggest[0]:
            biggest = [value, label]

    changes = {label: 1 if label == biggest[1] else 0 for label in labels}
    output = sitk.ChangeLabel(components, changeMap=changes)
    return output


def otsu_threshold(values, bins='auto'):
    hist, bin_edges = np.histogram(values, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


def segment_lungs(image, lower_threshold=-940, max_ratio=100, downscale=True):
    """Generate a segmentation mask for the lungs

    Parameters
    ----------
    image : str or SimpleITK.Image
        The thoracic CT image to segment or the path where it is stored
    lower_threshold : int
        The lower threshold for parenchymal tissue intensity - this helps differentiate air pockets from lung tissue (the default is -940)
    max_ratio : int
        The maximum ratio for the two lung sizes - this is used in case the lungs are not separated for the connected components step (the default is 100)
    downscale : bool
        If true, the image will be downscaled by half before segmentation to speed up the operation, but a full-size result will be returned (the default is False)
    """
    if isinstance(image, str):
        image = io.read_image(image)
    elif not isinstance(image, sitk.Image):
        raise ValueError("image must either be a path to an image or a SimpleITK.Image")

    image = sitk.Median(image)  # median filter in case of noise

    if downscale:
        src_image = image
        image = sitk.RecursiveGaussian(image, 3)
        image = apply_scaling(src_image, scaling_factor=0.5, interp=sitk.sitkBSpline)

    # Mask out the background so only the body is left
    body_mask = mask_body(image)

    if False:
        # Old Version
        # Find the threshold between body and lung intensitites
        clipped_image = clip_image(image)
        image_arr = sitk.GetArrayFromImage(clipped_image)
        bg_mask = 1 - sitk.GetArrayFromImage(body_mask)
        masked_image = np.ma.array(image_arr, mask=bg_mask)
        masked_values = masked_image.compressed()
        best_thresh = otsu_threshold(masked_values)
        del clipped_image
        del image_arr
        del masked_image

        # Initial threshold of the lungs
        thresh_image = sitk.BinaryThreshold(image, lower_threshold, best_thresh, 1, 0)
        and_filter = sitk.AndImageFilter()
        lung_mask = and_filter.Execute(body_mask, thresh_image)
        lung_mask = sitk.BinaryMorphologicalOpening(lung_mask, 1, sitk.sitkBall)
    else:
        masked_image = sitk.Mask(image, body_mask)
        thresh = sitk.OtsuMultipleThresholds(masked_image, numberOfThresholds=2, numberOfHistogramBins=256)
        lung_mask = sitk.Equal(thresh, 0)
        lung_mask = sitk.BinaryFillhole(lung_mask)

    # Remove any stray components and close holes
    components = sitk.ConnectedComponent(lung_mask)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(components)
    labels = lfilter.GetLabels()
    values = np.zeros(len(labels))
    for label in labels:
        values[label - 1] = lfilter.GetNumberOfPixels(label)
    keep_idx = np.argsort(values)[-2:]
    if len(keep_idx) > 1:
        ratio = values[keep_idx[0]] / values[keep_idx[1]]
        if ratio > max_ratio:
            keep_idx = [keep_idx[0]]
        elif ratio < (1 / max_ratio):
            keep_idx = [keep_idx[1]]
    to_keep = [labels[i] for i in keep_idx]
    changes = {label: 1 if label in to_keep else 0 for label in labels}
    output = sitk.ChangeLabel(components, changeMap=changes)

    if downscale:
        output = sitk.BinaryMorphologicalClosing(output, 1, sitk.sitkBall)
        output = apply_scaling(output, scaling_template=src_image, interp=sitk.sitkNearestNeighbor)
    else:
        output = sitk.BinaryMorphologicalClosing(output, 5, sitk.sitkBall)
    return sitk.Cast(output, 1)


def lung_connected_components(image, max_ratio=100):
    components = sitk.ConnectedComponent(image)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(components)
    labels = lfilter.GetLabels()
    values = np.zeros(len(labels))
    for label in labels:
        values[label - 1] = lfilter.GetNumberOfPixels(label)
    keep_idx = np.argsort(values)[-2:]

    if len(keep_idx) > 1:
        ratio = values[keep_idx[0]] / values[keep_idx[1]]
        if ratio > max_ratio:
            keep_idx = [keep_idx[0]]
        elif ratio < (1 / max_ratio):
            keep_idx = [keep_idx[1]]
    to_keep = [labels[i] for i in keep_idx]
    changes = {label: 1 if label in to_keep else 0 for label in labels}
    output = sitk.ChangeLabel(components, changeMap=changes)
    return output


def split_airways_from_lungmask(src_image, lung_mask):
    median_image = sitk.Median(src_image)
    clipped = clip_image(median_image, -1100, -850)
    masked_image = sitk.Mask(clipped, lung_mask)
    thresh_image = sitk.OtsuMultipleThresholds(masked_image, numberOfThresholds=4, numberOfHistogramBins=256)
    airways = sitk.Equal(thresh_image, 0)
    airways = get_biggest_object(airways)
    airways = sitk.Cast(airways, 1)
    inverse_airways = sitk.Not(airways)
    lungmask_noair = sitk.And(lung_mask, inverse_airways)
    lungmask_noair = sitk.BinaryMorphologicalOpening(lungmask_noair, 1, sitk.sitkBall)
    lungmask_noair = lung_connected_components(lungmask_noair)
    return lungmask_noair, airways


def compare_images(image1, image2):
    diff = sitk.Subtract(image1, image2)
    diff = sitk.Abs(diff)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(diff)
    print("Total Difference: {}".format(stats.GetSum()))
    print("Max Difference: {}".format(stats.GetMaximum()))


def get_nodule_overlap_stats(lung_mask, nodule_mask):
    lung_mask = sitk.Greater(lung_mask, 0)
    nodule_labels = sitk.ConnectedComponent(nodule_mask)
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(nodule_labels)
    nodule_stat = sitk.LabelShapeStatisticsImageFilter()
    results = []
    for label in stats_filter.GetLabels():
        nodule_pix = stats_filter.GetNumberOfPixels(label)
        single_nodule = sitk.Equal(nodule_labels, label)
        overlap = sitk.And(lung_mask, single_nodule)
        nodule_stat.Execute(overlap)
        overlap_pix = 0 if not nodule_stat.HasLabel(1) else nodule_stat.GetNumberOfPixels(1)
        centroid = nodule_mask.TransformPhysicalPointToIndex(stats_filter.GetCentroid(label))
        result = {
            'overlap_pixels': overlap_pix,
            'nodule_pixels': nodule_pix,
            'coverage_percent': overlap_pix / nodule_pix,
            'centroid_slice': centroid[2],
            'centroid_x': centroid[1],
            'centroid_y': centroid[0],
            'eq_sphere_radius_mm': stats_filter.GetEquivalentSphericalRadius(label)
        }
        results.append(result)
    return results


def get_overlap_stats(lung_pred, lung_label):
    lung_pred = sitk.Greater(lung_pred, 0)
    overlap_stat = sitk.LabelOverlapMeasuresImageFilter()
    overlap_stat.Execute(lung_pred, lung_label)
    dice = overlap_stat.GetDiceCoefficient()
    jaccard = overlap_stat.GetJaccardCoefficient()
    return dice, jaccard


def get_bounds(mask):
    """Get the corners of the bounding box/cube for the given mask"""
    is_image = False
    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)
        is_image = True
    elif isinstance(mask, SmartImage):
        mask = mask.as_array()
        is_image = True
    bounds = []
    for i in range(mask.ndim):
        axis_check = np.any(mask, axis=tuple([j for j in range(mask.ndim) if j != i]))
        axis_range = np.where(axis_check == True) # noqa
        bounds.append(slice(axis_range[0][0], axis_range[0][-1]))
    if len(bounds) == 3 and is_image:
        bounds = bounds[::-1]
    return bounds


def get_shared_bounds(mask1, mask2):
    bounds1, bounds2 = get_bounds(mask1), get_bounds(mask2)
    shared_bounds = [slice(max(bounds1[i].start, bounds2[i].start), min(bounds1[i].stop, bounds2[i].stop), None) for i in range(len(bounds1))]
    return shared_bounds


def get_label_bounds(mask, background_val=0, results_as_str=False):
    """Get the corners of the bounding box/cube for each label in a given mask

    Parameters
    ----------
    mask : SimpleITK.Image or SmartImage or numpy.ndarray
        The mask to analyze
    background_val : int
        The value of the background - bounds will not be returned for this label
    results_as_str : bool
        If true, return the bounds as "start_idx stop_idx" for each dimension separated by spaces

    Returns
    -------
    A dictionary with the segmentation label as keys and the bounds as values
    """
    is_image = False
    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)
        is_image = True
    elif isinstance(mask, SmartImage):
        mask = mask.as_array()
        is_image = True

    labels = np.unique(mask)
    results = {}
    for label in labels:
        if label == background_val:
            continue
        bounds = []
        label_mask = mask == label
        for i in range(mask.ndim):
            axis_check = np.any(label_mask, axis=tuple([j for j in range(mask.ndim) if j != i]))
            axis_range = np.where(axis_check == True) # noqa
            bounds.append(slice(axis_range[0][0], axis_range[0][-1]))
        if len(bounds) == 3 and is_image:
            bounds = bounds[::-1]
        if results_as_str:
            bounds = ' '.join(['{} {}'.format(item.start, item.stop) for item in bounds])
        results[label] = bounds
    return results

# ## ITK Methods ###


def convert_sitk_to_itk(image):
    if isinstance(image, itk.Image):
        return image
    itk_image = itk.GetImageFromArray(sitk.GetArrayViewFromImage(image), is_vector=image.GetNumberOfComponentsPerPixel() > 1)
    itk_image.SetOrigin(image.GetOrigin())
    itk_image.SetSpacing(image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [image.GetDimension()] * 2)))
    return itk_image


def convert_itk_to_sitk(image):
    if isinstance(image, sitk.Image):
        return image
    sitk_image = sitk.GetImageFromArray(itk.GetArrayViewFromImage(image), isVector=image.GetNumberOfComponentsPerPixel() > 1)
    sitk_image.SetOrigin(tuple(image.GetOrigin()))
    sitk_image.SetSpacing(tuple(image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return sitk_image


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


def get_distance_metrics(label_image, pred_image, label=None):
    if label is not None:
        label_image = sitk.Equal(label_image, int(label))
        pred_image = sitk.Equal(pred_image, int(label))

    label_dist = sitk.SignedMaurerDistanceMap(label_image, squaredDistance=False, useImageSpacing=True)
    label_surf = sitk.LabelContour(label_image)

    pred_dist = sitk.SignedMaurerDistanceMap(pred_image, squaredDistance=False, useImageSpacing=True)
    pred_surf = sitk.LabelContour(pred_image)

    label_surf_arr = sitk.GetArrayViewFromImage(label_surf) > 0.5
    pred_surf_arr = sitk.GetArrayViewFromImage(pred_surf) > 0.5

    pred2label_arr = sitk.GetArrayViewFromImage(label_dist)[pred_surf_arr]
    label2pred_arr = sitk.GetArrayViewFromImage(pred_dist)[label_surf_arr]

    all_dist_arr = np.concatenate([label2pred_arr, pred2label_arr])

    mean_abs_pred_dist = np.mean(np.abs(pred2label_arr))
    mean_rel_pred_dist = np.mean(pred2label_arr)
    mean_abs_s2s_dist = np.mean(np.abs(all_dist_arr))
    median_abs_s2s_dist = np.median(np.abs(all_dist_arr))
    max_abs_s2s_dist = np.max(np.abs(all_dist_arr))
    results = {
        'MeanAbsolutePredictionDistance': mean_abs_pred_dist,
        'MeanRelativePredictionDistance': mean_rel_pred_dist,
        'MeanSurfaceToSurfaceDistance': mean_abs_s2s_dist,
        'MedianSurfaceToSurfaceDistance': median_abs_s2s_dist,
        'MaxSurfaceToSurfaceDistance': max_abs_s2s_dist,
    }
    return results


def add_label_to_empty(label1, label2):
    """Add a new binary label to an existing one

     Parameters
     ----------
     label1 : sitk.Image
        The base label to add things to - can have multiple values
    label2 : sitk.Image
        The new label to add to the base - must have values in [0, 1]
     """
    arr1 = sitk.GetArrayViewFromImage(label1)
    arr2 = sitk.GetArrayViewFromImage(label2)
    new_label = arr1.max() + 1
    result = np.where(arr1 == 0, arr2 * new_label, arr1)
    result_label = sitk.GetImageFromArray(result)
    result_label.CopyInformation(label1)
    return result_label


def zeros_like(image, dtype=None):
    if not isinstance(image, dict):
        image = get_sampling_info(image)
    if dtype is not None:
        image['dtype'] = dtype
    zero_image = sitk.Image(image['size'], image['dtype'])
    zero_image.SetOrigin(image['origin'])
    zero_image.SetSpacing(image['spacing'])
    zero_image.SetDirection(image['direction'])
    if isinstance(image, SmartImage):
        zero_image = SmartImage(zero_image)
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
    if isinstance(label_img, SmartImage):
        label_img = label_img.sitk_image
    components = sitk.ConnectedComponent(label_img)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(components)
    labels = lfilter.GetLabels()
    changes = {}
    for label in labels:
        changes[label] = 0 if lfilter.GetNumberOfPixels(label) < min_size else 1
    result = sitk.ChangeLabel(components, changeMap=changes)
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


@wrap4numpy
def argmax(image, axis=-1, return_type=np.uint8):
    return image.argmax(axis=axis).astype(return_type)
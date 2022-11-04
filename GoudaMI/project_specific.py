"Methods that are too project specific to be put with the rest, or use extra libraries (ie. scipy, opencv, etc.)"
import os
import warnings
from typing import Any

import numpy as np
import SimpleITK as sitk

from GoudaMI.constants import MIN_INTENSITY_PULMONARY_CT
from GoudaMI.convert import as_array
from GoudaMI.ct_utils import resample_to_ref, resample_rescale
from GoudaMI.smart_image import SmartImage


def quick_save(data: Any, label=None, skip_existing_filenames=False):
    """Quickly save an image/label in the project's scratch folder
    Used in quick debugging with iterative output filenames

    Parameters
    ----------
    data: array_like
        The data to save
    label: str
        The full or extension-less filename to save the data under (the default is None)
    skip_existing_filenames: bool
        Whether to skip saving the file if the label is already taken (the default is False)
    """
    import gouda
    scratch_dir = gouda.GoudaPath(os.getcwd() + '/scratch').ensure_dir()
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


def resample_isocube_to_source(cube_path, cropped_path=None, original_path=None, label=None, outside_val=MIN_INTENSITY_PULMONARY_CT):
    """Used in lung segmentation to convert a isotropic cube back to the source parameters"""
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


def clean_segmentation(lung_image, lung_threshold=0.1, body_mask=None):
    """Used in lung segmentation to clean the output segmentation"""
    import skimage.measure
    source_arr = as_array(lung_image)
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


def resample_to_isocube(image, output_size=300, interp=sitk.sitkBSpline, outside_val=-1000):
    """Resize an image to have equal spacing and sizing in each axis, and pad
    with background value to preserve aspect ratio

    NOTE - in retrospect, enforcing spacing and size means the image is getting stretched and squashed. Best not to use this one.
    """
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


def segment_lungs(image, lower_threshold=-940, max_ratio=100, downscale=True):
    """Generate a segmentation mask for the lungs based on intensity values

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
    from .io import read_image
    from .ct_utils import mask_body, clip_image, otsu_threshold
    if isinstance(image, str):
        image = read_image(image)
    elif not isinstance(image, sitk.Image):
        raise ValueError("image must either be a path to an image or a SimpleITK.Image")

    image = sitk.Median(image)  # median filter in case of noise

    if downscale:
        src_image = image
        image = sitk.RecursiveGaussian(image, 3)
        image = resample_rescale(src_image, scaling_factor=0.5, interp=sitk.sitkBSpline)

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
    lfilter.SetComputePerimeter(False)
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
        output = resample_rescale(output, scaling_template=src_image, interp=sitk.sitkNearestNeighbor)
    else:
        output = sitk.BinaryMorphologicalClosing(output, 5, sitk.sitkBall)
    return sitk.Cast(output, 1)


def lung_connected_components(image, max_ratio=10):
    """Connected components pruning, but keep either the largest object or largest 2 objects if they are close in size"""
    components = sitk.ConnectedComponent(image)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.SetComputePerimeter(False)
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
    """Use thresholding to approximate the airways given an image and lung mask"""
    from .ct_utils import clip_image, get_largest_object
    median_image = sitk.Median(src_image)
    clipped = clip_image(median_image, -1100, -850)
    masked_image = sitk.Mask(clipped, lung_mask)
    thresh_image = sitk.OtsuMultipleThresholds(masked_image, numberOfThresholds=4, numberOfHistogramBins=256)
    airways = sitk.Equal(thresh_image, 0)
    airways = get_largest_object(airways)
    airways = sitk.Cast(airways, 1)
    inverse_airways = sitk.Not(airways)
    lungmask_noair = sitk.And(lung_mask, inverse_airways)
    lungmask_noair = sitk.BinaryMorphologicalOpening(lungmask_noair, 1, sitk.sitkBall)
    lungmask_noair = lung_connected_components(lungmask_noair)
    return lungmask_noair, airways


def array_get_surface(image, connectivity=1):
    import scipy.ndimage.morphology
    image = image.astype(np.bool)
    conn = scipy.ndimage.morphology.generate_binary_structure(image.ndim, connectivity)
    return image ^ scipy.ndimage.morphology.binary_erosion(image, conn)


def array_get_distances(image_1, image_2, sampling=1, connectivity=1):
    import scipy.ndimage.morphology
    surf_1 = array_get_surface(image_1, connectivity=connectivity)
    surf_2 = array_get_surface(image_2, connectivity=connectivity)
    dta = scipy.ndimage.morphology.distance_transform_edt(~surf_1, sampling)
    dtb = scipy.ndimage.morphology.distance_transform_edt(~surf_2, sampling)
    return np.concatenate([np.ravel(dta[surf_2 != 0]), np.ravel(dtb[surf_1 != 0])])


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


def fill2d(arr, force_slices=False):
    # TODO - update this to not require scipy
    import scipy.ndimage.morphology
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
    # TODO - update this to not require scipy
    import scipy.ndimage.morphology
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
from typing import Tuple, Union
import warnings

import gouda
import numpy as np
import SimpleITK as sitk

from GoudaMI.ct_utils import get_unique_labels, get_shared_bounds
from GoudaMI.smart_image import SmartImage, ImageType, as_image, get_image_type


def get_centroid_metrics(label_image, pred_image, return_centroids=False, return_sizes=False):
    """Return centroid distances between each object in each label.

    Note
    ----
    distances are indexed by [label val - 1, pred val - 1] - so if the value of the object in the label image is 2 and the value is 4 in the predicted image, the distance between the two is at dist_arr[1, 3] (since 0 is the bg value, we skip it)
    """
    if get_image_type(label_image) != 'sitk':
        label_image = as_image(label_image).sitk_image
    if get_image_type(pred_image) != 'sitk':
        pred_image = as_image(pred_image).sitk_image
    filter1 = sitk.LabelShapeStatisticsImageFilter()
    filter1.Execute(label_image)

    filter2 = sitk.LabelShapeStatisticsImageFilter()
    filter2.Execute(pred_image)

    label_centroids = {}
    label_sizes = {}
    pred_centroids = {}
    pred_sizes = {}
    for label in filter1.GetLabels():
        label_centroids[label] = np.array(filter1.GetCentroid(label))
        label_sizes[label] = filter1.GetPhysicalSize(label)
    for label in filter2.GetLabels():
        pred_centroids[label] = np.array(filter2.GetCentroid(label))
        pred_sizes[label] = filter2.GetPhysicalSize(label)

    dist_arr = np.zeros([max(label_centroids.keys()),
                        max(pred_centroids.keys())])
    for label_key in label_centroids:
        for pred_key in pred_centroids:
            dist_arr[label_key - 1, pred_key - 1] = np.linalg.norm(
                label_centroids[label_key] - pred_centroids[pred_key])

    to_return = [dist_arr]
    if return_centroids:
        to_return.extend([label_centroids, pred_centroids])
    if return_sizes:
        to_return.extend([label_sizes, pred_sizes])
    return to_return


def get_comparison_metrics(label_image, pred_image, overlap_metrics=True, distance_metrics=True, fully_connected=True, surface_dice=True, nsd_tol=2, labels=None):
    """Get comparison metrics between a label and a predicted label

    Parameters
    ----------
    label_image : ImageType
        The label image (generally independent standard or ground truth)
    pred_image : ImageType
        The predicted label image
    overlap_metrics : bool, optional
        Whether to compute overlap metrics (Dice, Jaccard, FP, FN), by default True
    distance_metrics : bool, optional
        Whether to compute distance metrics (ASSD, RSSD, Hausdorff, Hausdorff95), by default True
    fully_connected : bool, optional
        In distance metrics - whether connected components are defined by face connectivity (False) or face+edge+vertex connectivity (True), by default True
    surface_dice : bool, optional
        Whether to compute Normalized Surface Dice, by default True
    nsd_tol : int | list, optional
        Tolerance value(s) to use for computing Normalized Surface Dice in physical units, should be listed based on ascending order of label indices being compared, by default 2
    labels : list, optional
        The subset of label values to evaluate if you don't want to evaluate all non-zero labels, by default None
    """
    if get_image_type(label_image) != 'sitk':
        label_image = as_image(label_image).sitk_image
    if get_image_type(pred_image) != 'sitk':
        pred_image = as_image(pred_image).sitk_image

    pred_stats = sitk.LabelShapeStatisticsImageFilter()
    pred_stats.Execute(pred_image)
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(label_image)

    if labels is None:
        labels = label_stats.GetLabels()

    nsd_tol = gouda.force_len(nsd_tol, len(labels))

    image_results = {}
    if overlap_metrics:
        overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_filter.Execute(pred_image, label_image)
    for label_idx, label in enumerate(labels):
        image_results[label] = {}
        if overlap_metrics:
            overlap_results = {
                'Dice': float(overlap_filter.GetDiceCoefficient(int(label))),
                'Jaccard': float(overlap_filter.GetJaccardCoefficient(int(label))),
                'FalsePositive': float(overlap_filter.GetFalsePositiveError(int(label))),
                'FalseNegative': float(overlap_filter.GetFalseNegativeError(int(label)))
            }
            for key in overlap_results:
                image_results[label][key] = overlap_results[key]
        if distance_metrics:
            if label not in pred_stats.GetLabels():
                warnings.warn('Missing label in predicted image - defaulting distance to nan')
                dist_results = {'ASSD': np.nan,
                                'RSSD': np.nan,
                                'HausdorffDistance': np.nan,
                                'HausdorffDistance95': np.nan}
            else:
                dist_results = get_binary_distance_metrics(
                    label_image, pred_image,
                    label=int(label), fully_connected=fully_connected)
            for key in dist_results:
                image_results[label][key] = float(dist_results[key])
        if surface_dice:
            if label not in pred_stats.GetLabels():
                warnings.warn('Missing label in predicted image - defaulting surface dice to 0')
                nsd = 0
            else:
                nsd = normalized_surface_dice(label_image == label, pred_image == label, tol=nsd_tol[label_idx])
            image_results[label]['NSD'] = float(nsd)
    return image_results


def get_binary_distance_metrics(label_image, pred_image, fully_connected=True, label=None, detailed=False):
    """Get distance-based metrics between the label and predicted label

    Parameters
    ----------
    label_image : SimpleITK.Image
        The ground truth label image
    pred_image : SimpleITK.Image
        The predicted label image
    label : int or None
        The label of interest to find metrics for - if None, assumes the labels are already binary
    detailed : bool
        If True, also finds mean, median, and max surface-to-surface distances
    """
    if isinstance(label_image, SmartImage):
        label_image = label_image.sitk_image
    if isinstance(pred_image, SmartImage):
        pred_image = pred_image.sitk_image

    if label is not None:
        label_image = sitk.Equal(label_image, int(label))
        pred_image = sitk.Equal(pred_image, int(label))

    # Crop to the shared bounds to speed up computation
    shared_bounds = get_shared_bounds(label_image, pred_image, as_slice=True, padding=3)
    label_image = label_image[shared_bounds]
    pred_image = pred_image[shared_bounds]

    # Add a zero padding - it messes up the contours if the label is on an edge
    label_image = sitk.ConstantPad(label_image, [1, 1, 1], [1, 1, 1], 0)
    pred_image = sitk.ConstantPad(pred_image, [1, 1, 1], [1, 1, 1], 0)

    label_dist = sitk.SignedMaurerDistanceMap(
        label_image, squaredDistance=False, useImageSpacing=True)
    label_surf = sitk.LabelContour(label_image, fullyConnected=fully_connected)

    pred_dist = sitk.SignedMaurerDistanceMap(
        pred_image, squaredDistance=False, useImageSpacing=True)
    pred_surf = sitk.LabelContour(pred_image, fullyConnected=fully_connected)

    label_surf_arr = sitk.GetArrayViewFromImage(label_surf).astype(bool)
    pred_surf_arr = sitk.GetArrayViewFromImage(pred_surf).astype(bool)
    if label_surf_arr.sum() == 0 or pred_surf_arr.sum() == 0:
        mean_abs_pred_dist = np.nan
        mean_rel_pred_dist = np.nan
        haus_dist = np.nan
        haus_95 = np.nan
        all_dist_arr = None
    else:
        pred2label_arr = sitk.GetArrayViewFromImage(label_dist)[pred_surf_arr]
        label2pred_arr = sitk.GetArrayViewFromImage(pred_dist)[label_surf_arr]

        all_dist_arr = np.concatenate([label2pred_arr, pred2label_arr])

        # Distance from each prediction voxel to the nearest label voxel
        mean_abs_pred_dist = np.mean(np.abs(pred2label_arr))

        # Signed distance from each prediction voxel to the nearest label voxel
        # Inside the label is negative
        mean_rel_pred_dist = np.mean(pred2label_arr)

        # Distance from each prediction voxel to the nearest label voxel and from each label voxel to the nearest prediction voxel
        # Max is similar to Hausdorff, but includes predicted boundary inside of label boundary as error
        haus_dist = np.max(np.maximum(all_dist_arr, 0))
        haus_95 = np.percentile(np.maximum(all_dist_arr, 0), [95])[0]

    results = {
        'ASSD': mean_abs_pred_dist,  # AKA - mean absolute prediction distance
        'RSSD': mean_rel_pred_dist,  # AKA - mean relative prediction distance
        'HausdorffDistance': haus_dist,
        'HausdorffDistance95': haus_95,
    }
    if detailed:
        if all_dist_arr is None:
            warnings.warn('No surface voxels found - defaulting distance to nan')
            results['MeanSurfaceToSurfaceDistance'] = np.nan
            results['MedianSurfaceToSurfaceDistance']: np.nan
            results['MaxSurfaceToSurfaceDistance'] = np.nan
        else:
            results['MeanSurfaceToSurfaceDistance'] = np.mean(np.abs(all_dist_arr))
            results['MedianSurfaceToSurfaceDistance']: np.median(np.abs(all_dist_arr))
            results['MaxSurfaceToSurfaceDistance'] = np.max(np.abs(all_dist_arr))
    return results


def get_object_comparison_metrics(true_label, pred_label, dtype=sitk.sitkUInt8, result_extras=None):
    """Get comparison metrics for each object in the true label

    Parameters
    ----------
    true_label : sitk.Image, SmartImage
        The ground truth binary label image
    pred_label : sitk.Image, SmartImage
        The predicted binary label image
    dtype : int, optional
        The dtype to use for the labels/connected components, by default sitk.sitkUInt8 (use sitk.sitkUInt16 if >255 objects)
    result_extras : dict, optional
        Extra key/value pairs to put in each result in the list (for dataframes), by default None

    Returns
    -------
    list
        List where each item is a dict with results for an object
    """
    if hasattr(true_label, 'sitk_image'):
        # if isinstance(true_label, SmartImage):
        true_label = true_label.sitk_image
    if hasattr(pred_label, 'sitk_image'):
        # if isinstance(pred_label, SmartImage):
        pred_label = pred_label.sitk_image
    result_extras = {} if result_extras is None else result_extras

    objects = sitk.Cast(sitk.ConnectedComponent(true_label), dtype)
    pred_objects = sitk.Cast(sitk.ConnectedComponent(pred_label), dtype)
    overlap_objects = objects * pred_label

    true_object_filter = sitk.LabelShapeStatisticsImageFilter()
    true_object_filter.SetBackgroundValue(0)
    true_object_filter.SetComputePerimeter(False)
    true_object_filter.Execute(objects)
    object_labels = true_object_filter.GetLabels()

    overlap_object_filter = sitk.LabelShapeStatisticsImageFilter()
    overlap_object_filter.SetBackgroundValue(0)
    overlap_object_filter.SetComputePerimeter(False)
    overlap_object_filter.Execute(overlap_objects)

    overlap_object_labels = overlap_object_filter.GetLabels()

    overlap_metrics = []
    for label_idx in object_labels:
        # Get only one object at a time
        if label_idx not in overlap_object_labels:
            object_result = {}
        else:
            label_object = objects == label_idx

            # Any guessed fg object that overlaps with the true fg object
            guessed_overlap = label_object * pred_objects
            unique_guessed = get_unique_labels(guessed_overlap)[1:]
            guessed_object = pred_objects == unique_guessed[0]
            for object_idx in unique_guessed[1:]:
                guessed_object = guessed_object + pred_objects == object_idx

            object_result = get_comparison_metrics(
                label_object, guessed_object)[1]
            object_result['Coverage'] = overlap_object_filter.GetPhysicalSize(
                label_idx) / true_object_filter.GetPhysicalSize(label_idx)
        object_result['ObjectSize'] = true_object_filter.GetPhysicalSize(
            label_idx)
        object_result['Elongation'] = true_object_filter.GetElongation(
            label_idx)

        for key in result_extras:
            object_result[key] = result_extras[key]
        object_result['ObjectIdx'] = label_idx

        overlap_metrics.append(object_result)
    return overlap_metrics


def bilateral_overlap_stats(label1: ImageType, label2: ImageType) -> Tuple[dict, dict]:
    """Get overlap stats between two images

    Parameters
    ----------
    label1 : SimpleITK.Image, SmartImage
        The first binary label to compare
    label2 : SimpleITK.Image, SmartImage
        The second binary label to compare

    Returns
    -------
        A tuple of dicts for the results relative to the first label and second label (in that order)
    """
    label1 = SmartImage(label1)
    label2 = SmartImage(label2)
    label1_size = label1.volume()
    label2_size = label2.volume()

    union_label = label1.binary_and(label2)
    union_size = union_label.volume()

    solo1 = label1 - union_label
    solo1_size = solo1.volume()
    solo2 = label2 - union_label
    solo2_size = solo2.volume()

    results = [
        {'size': label1_size,
         'exclusive_size': solo1_size,
         'exclusive_ratio': solo1_size / label1_size,
         'shared_size': union_size,
         'shared_ratio': union_size / label1_size
         },
        {'size': label2_size,
         'exclusive_size': solo2_size,
         'exclusive_ratio': solo2_size / label2_size,
         'shared_size': union_size,
         'shared_ratio': union_size / label2_size
         }
    ]
    return results


def normalized_surface_dice(label1: sitk.Image, label2: sitk.Image, tol: float = 2):
    """Return the NSD between two 3D binary labels

    Parameters
    ----------
    label1 : sitk.Image
        First binary label image
    label2 : sitk.Image
        Second binary label image
    tol : float | list
        The tolerated difference between boundaries in physical units, by default 2
    kwargs : dict
        Optional keyword arguments to pass to :func:`GoudaMI.measure.get_distances`


    NOTE
    ----
    The Normalized Surface Dice (aka Normalized Surface Distance, Surface Dice) is an uncertainty-aware measure used to compare the overlap between two segmentation surfaces. Rather than using voxel overlap, it measures the overlap of the surfaces within a tolerance distance of eachother. This tolerance can be based on domain-requirements or inter-observer variability.
    Original Pub: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8314151/
    Implementation adapted from: https://github.com/deepmind/surface-distance/

    Some recommended tolerances in CT images (from the FLARE22 competition):
        * Adrenal Gland (Left/Right)- 2mm
        * Aorta - 2mm
        * Duodenum - 7mm
        * Esophagus - 3mm
        * Gallbladder - 2mm
        * Inferior Vena Cava - 2mm
        * Kidney (Right/Left) - 3mm
        * Liver - 5mm
        * Pancreas - 5mm
        * Spleen - 3mm
        * Stomach - 5mm

    More recommended tolerance (Medical Segmenation Decathlon):
        * Brain (edema, enhancing & non-enhancing tumor) - multimodal MRI - 5mm
        * Heart (left atrium) - MRI - 4mm
        * Hippocampus (anterior and posterior) - MRI - 1mm
        * Liver - portal venous phase CT - 7mm
        * Lung Nodule - CT - 2mm
        * Prostate (prosate peripheral zone and transition zone) - T2 MRI - 4mm
        * Pancreas (pancreatic parenchyma and cyst/tumor) - porrtal venous phase CT - 5mm
        * Colon (colon cancer primaries) - CT - 4mm
        * Hepatic vessel (vessel and tumor) - CT - 3mm
        * Spleen - CT - 3mm
    """
    from ._measure_deepmind import (
        ENCODE_NEIGHBOURHOOD_3D_KERNEL,
        create_table_neighbour_code_to_surface_area)
    label1 = as_image(label1).sitk_image
    label2 = as_image(label2).sitk_image

    # crop to a shared bounding box to reduce computation
    bounds = get_shared_bounds(label1, label2, as_slice=True, padding=5)
    label1 = label1[bounds]
    label2 = label2[bounds]

    spacing = np.asarray(label1.GetSpacing())
    spacing = spacing[::-1]
    neighborhood_code = create_table_neighbour_code_to_surface_area(spacing)
    sitk_kernel = sitk.GetImageFromArray(
        ENCODE_NEIGHBOURHOOD_3D_KERNEL.astype(np.uint8))
    full_true_neighbors = 0b11111111

    gt_code_map = sitk.Convolution(label1, sitk_kernel, boundaryCondition=0)
    gt_code_map_arr = sitk.GetArrayViewFromImage(gt_code_map)
    gt_surface_area_map = neighborhood_code[gt_code_map_arr]
    gt_borders = sitk.And(gt_code_map != 0, gt_code_map != full_true_neighbors)
    gt_surfel_area = gt_surface_area_map[sitk.GetArrayViewFromImage(
        gt_borders).astype(np.bool_)]
    gt_dist_map = sitk.SignedMaurerDistanceMap(gt_borders, squaredDistance=False, useImageSpacing=True, insideIsPositive=False)

    pred_code_map = sitk.Convolution(label2, sitk_kernel, boundaryCondition=0)
    pred_code_map_arr = sitk.GetArrayViewFromImage(pred_code_map)
    pred_surface_area_map = neighborhood_code[pred_code_map_arr]
    pred_borders = sitk.And(pred_code_map != 0, pred_code_map != full_true_neighbors)
    pred_surfel_area = pred_surface_area_map[sitk.GetArrayViewFromImage(
        pred_borders).astype(np.bool_)]
    pred_dist_map = sitk.SignedMaurerDistanceMap(pred_borders, squaredDistance=False, useImageSpacing=True, insideIsPositive=False)

    gt_dist = sitk.GetArrayViewFromImage(gt_dist_map)
    gt_bord = sitk.GetArrayViewFromImage(gt_borders).astype(np.bool_)
    pred_dist = sitk.GetArrayViewFromImage(pred_dist_map)
    pred_bord = sitk.GetArrayViewFromImage(pred_borders).astype(np.bool_)

    dist_gt2pred = pred_dist[gt_bord]
    dist_pred2gt = gt_dist[pred_bord]

    if not gouda.is_iter(tol):
        gt_overlap = np.sum(gt_surfel_area[dist_gt2pred <= tol])
        pred_overlap = np.sum(pred_surfel_area[dist_pred2gt <= tol])
        gt_surf_area = np.sum(gt_surfel_area)
        pred_surf_area = np.sum(pred_surfel_area)
        return (gt_overlap + pred_overlap) / (gt_surf_area + pred_surf_area)
    else:
        results = []
        for tol_item in tol:
            gt_overlap = np.sum(gt_surfel_area[dist_gt2pred <= tol_item])
            pred_overlap = np.sum(pred_surfel_area[dist_pred2gt <= tol_item])
            gt_surf_area = np.sum(gt_surfel_area)
            pred_surf_area = np.sum(pred_surfel_area)
            results.append((gt_overlap + pred_overlap) / (gt_surf_area + pred_surf_area))
        return results


def get_distances(label1: sitk.Image, label2: sitk.Image, direction: str = 'both', use_squared_distance: bool = False, use_image_spacing: bool = True, fully_connected_contours: bool = False):
    """Return the distances from all points on one label's surface to the other and vis-versa

    Parameters
    ----------
    label1 : sitk.Image
        The first label image to use
    label2 : sitk.Image
        The second label image to use
    direction : str, optional
        'both' to get distances from label1 to label2 and vis-versa or 'single' to get label2's distance from label1 only, by default 'both'
    use_squared_distance : bool, optional
        Whether to use squared distances, by default False
    use_image_spacing : bool, optional
        Whether to use image spacing in distance measures, by default True
    fully_connected_contours : bool, optional
        Whether to use fully connected contours when creating surfaces, by default False

    Returns
    -------
    dists or [dists1, dists2]
        Distances from each point on label 2 to the nearest points on label 1 (optionally the same from label 1 to label 2 if direction is 'both')
    """
    if isinstance(label1, SmartImage):
        label1 = label1.sitk_image
    if isinstance(label2, SmartImage):
        label2 = label2.sitk_image

    label1 = sitk.ConstantPad(label1, [1, 1, 1], [1, 1, 1], 0)
    label2 = sitk.ConstantPad(label2, [1, 1, 1], [1, 1, 1], 0)

    label1_dist = sitk.SignedMaurerDistanceMap(
        label1, squaredDistance=use_squared_distance, useImageSpacing=use_image_spacing)
    label1_dist_arr = sitk.GetArrayViewFromImage(label1_dist)
    label2_surf = sitk.LabelContour(
        label2, fullyConnected=fully_connected_contours)
    label2_surf_arr = sitk.GetArrayViewFromImage(label2_surf).astype(bool)

    if direction == 'both':
        label2_dist = sitk.SignedMaurerDistanceMap(
            label2, squaredDistance=use_squared_distance, useImageSpacing=use_image_spacing)
        label2_dist_arr = sitk.GetArrayViewFromImage(label2_dist)
        label1_surf = sitk.LabelContour(
            label1, fullyConnected=fully_connected_contours)
        label1_surf_arr = sitk.GetArrayViewFromImage(label1_surf).astype(bool)

    dists1 = label1_dist_arr[label2_surf_arr]  # dist from label2 to label1
    if direction == 'both':
        dists2 = label2_dist_arr[label1_surf_arr]  # dist from label1 to label2
        return dists1, dists2
    return dists1

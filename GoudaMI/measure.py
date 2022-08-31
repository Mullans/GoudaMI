import numpy as np
import SimpleITK as sitk

from .smart_image import SmartImage
from .ct_utils import get_unique_labels


def get_centroid_metrics(label_image, pred_image, return_centroids=False, return_sizes=False):
    """Return centroid distances between each object in each label.

    Note
    ----
    distances are indexed by [label val - 1, pred val - 1] - so if the value of the object in the label image is 2 and the value is 4 in the predicted image, the distance between the two is at dist_arr[1, 3] (since 0 is the bg value, we skip it)
    """
    if isinstance(label_image, SmartImage):
        label_image = label_image.sitk_image
    if isinstance(pred_image, SmartImage):
        pred_image = pred_image.sitk_image
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

    dist_arr = np.zeros([max(label_centroids.keys()), max(pred_centroids.keys())])
    for label_key in label_centroids:
        for pred_key in pred_centroids:
            dist_arr[label_key - 1, pred_key - 1] = np.linalg.norm(label_centroids[label_key] - pred_centroids[pred_key])

    to_return = [dist_arr]
    if return_centroids:
        to_return.extend([label_centroids, pred_centroids])
    if return_sizes:
        to_return.extend([label_sizes, pred_sizes])
    return to_return


def get_comparison_metrics(label_image, pred_image, overlap_metrics=True, distance_metrics=True, fully_connected=True, labels=None):
    if isinstance(label_image, SmartImage):
        label_image = label_image.sitk_image
    if isinstance(pred_image, SmartImage):
        pred_image = pred_image.sitk_image
    if labels is None:
        label_filt = sitk.LabelShapeStatisticsImageFilter()
        label_filt.Execute(label_image)
        labels = label_filt.GetLabels()

    image_results = {}
    if overlap_metrics:
        overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_filter.Execute(pred_image, label_image)
    for label_idx in labels:
        image_results[label_idx] = {}
        if overlap_metrics:
            overlap_results = {
                'Dice': float(overlap_filter.GetDiceCoefficient(int(label_idx))),
                'Jaccard': float(overlap_filter.GetJaccardCoefficient(int(label_idx))),
                'FalsePositive': float(overlap_filter.GetFalsePositiveError(int(label_idx))),
                'FalseNegative': float(overlap_filter.GetFalseNegativeError(int(label_idx)))
            }
            for key in overlap_results:
                image_results[label_idx][key] = overlap_results[key]
        if distance_metrics:
            dist_results = get_binary_distance_metrics(label_image, pred_image, label=int(label_idx), fully_connected=fully_connected)
            for key in dist_results:
                image_results[label_idx][key] = float(dist_results[key])
    return image_results


def get_binary_distance_metrics(label_image, pred_image, fully_connected=False, label=None):
    """Get distance-based metrics between the label and predicted label

    Parameters
    ----------
    label_image : SimpleITK.Image
        The ground truth label image
    pred_image : SimpleITK.Image
        The predicted label image
    label : int or None
        The label of interest to find metrics for - if None, assumes the labels are already binary
    """
    if isinstance(label_image, SmartImage):
        label_image = label_image.sitk_image
    if isinstance(pred_image, SmartImage):
        pred_image = pred_image.sitk_image

    if label is not None:
        label_image = sitk.Equal(label_image, int(label))
        pred_image = sitk.Equal(pred_image, int(label))

    # Add a zero padding - it messes up the contours if the label is on an edge
    label_image = sitk.ConstantPad(label_image, [1, 1, 1], [1, 1, 1], 0)
    pred_image = sitk.ConstantPad(pred_image, [1, 1, 1], [1, 1, 1], 0)

    # label_dist = sitk.SignedDanielssonDistanceMap(label_image, squaredDistance=False, useImageSpacing=True)
    label_dist = sitk.SignedMaurerDistanceMap(label_image, squaredDistance=False, useImageSpacing=True)
    label_surf = sitk.LabelContour(label_image, fullyConnected=fully_connected)

    # pred_dist = sitk.SignedDanielssonDistanceMap(pred_image, squaredDistance=False, useImageSpacing=True)
    pred_dist = sitk.SignedMaurerDistanceMap(pred_image, squaredDistance=False, useImageSpacing=True)
    pred_surf = sitk.LabelContour(pred_image, fullyConnected=fully_connected)

    label_surf_arr = sitk.GetArrayViewFromImage(label_surf).astype(bool)
    pred_surf_arr = sitk.GetArrayViewFromImage(pred_surf).astype(bool)
    if label_surf_arr.sum() == 0 or pred_surf_arr.sum() == 0:
        mean_abs_pred_dist = np.nan
        mean_rel_pred_dist = np.nan
        mean_abs_s2s_dist = np.nan
        median_abs_s2s_dist = np.nan
        max_abs_s2s_dist = np.nan
        haus_dist = np.nan
        haus_95 = np.nan

        # haus_sitk = np.nan
        # avg_haus_sitk = np.nan
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
        mean_abs_s2s_dist = np.mean(np.abs(all_dist_arr))
        median_abs_s2s_dist = np.median(np.abs(all_dist_arr))
        max_abs_s2s_dist = np.max(np.abs(all_dist_arr))
        haus_dist = np.max(np.maximum(all_dist_arr, 0))
        haus_95 = np.percentile(np.maximum(all_dist_arr, 0), [95])[0]

        ## Just to validate hausdorff
        # hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        # hausdorff_filter.Execute(label_image, pred_image)
        # haus_sitk = float(hausdorff_filter.GetHausdorffDistance())
        # avg_haus_sitk = float(hausdorff_filter.GetAverageHausdorffDistance())

    results = {
        # 'MeanAbsolutePredictionDistance': mean_abs_pred_dist,
        'ASSD': mean_abs_pred_dist,
        # 'MeanRelativePredictionDistance': mean_rel_pred_dist,
        'SSSD': mean_rel_pred_dist,
        'MeanSurfaceToSurfaceDistance': mean_abs_s2s_dist,
        'MedianSurfaceToSurfaceDistance': median_abs_s2s_dist,
        'MaxSurfaceToSurfaceDistance': max_abs_s2s_dist,
        'HausdorffDistance': haus_dist,
        'HausdorffDistance95': haus_95,
        # 'HausdorffDistance_SITK': haus_sitk,
        # 'AverageHausdorffDistance_SITK': avg_haus_sitk
    }
    return results


def get_object_comparison_metrics(true_label, pred_label, dtype=sitk.sitkUInt8, result_extras={}):
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
        Extra values to put in each result in the list (for dataframes), by default {}

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
            
            object_result = get_comparison_metrics(label_object, guessed_object)[1]
            object_result['Coverage'] = overlap_object_filter.GetPhysicalSize(label_idx) / true_object_filter.GetPhysicalSize(label_idx)
        object_result['ObjectSize'] = true_object_filter.GetPhysicalSize(label_idx)
        object_result['Elongation'] = true_object_filter.GetElongation(label_idx)
            
        for key in result_extras:
            object_result[key] = result_extras[key]
        object_result['ObjectIdx'] = label_idx

        overlap_metrics.append(object_result)
    return overlap_metrics
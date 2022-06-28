import urllib

import gouda
import SimpleITK as sitk

from .smart_image import SmartImage

try:
    sitk.ElastixImageFilter
except AttributeError:
    raise ImportError("These methods require the SimpleElastix extension to SimpleITK - check simpleelastix.github.io for more info.")

# https://github.com/SuperElastix/ElastixModelZoo/tree/master/models
PARAMETERS = {
    54: 'https://raw.githubusercontent.com/SuperElastix/ElastixModelZoo/master/models/Par0054/Par0054_sstvd.txt',
    58: 'https://raw.githubusercontent.com/SuperElastix/ElastixModelZoo/master/models/Par0058/Par0058trans.txt'
}


def get_elastix_params(param_number=54, param_dir='param_files', overwrite=False):
    if param_number not in PARAMETERS:
        raise ValueError('Only parameters {} have been set up.'.format(list(PARAMETERS.keys())))
    param_url = PARAMETERS[param_number]

    data_dir = gouda.GoudaPath(param_dir).ensure_dir()
    param_path = data_dir.add_basename(param_url)
    if param_path.exists() and not overwrite:
        return sitk.ReadParameterFile(param_path.abspath)
    attempts = 0
    max_attempts = 5
    success = False
    while attempts < max_attempts and not success:
        try:
            response = urllib.request.urlopen(param_url)
            success = True
        except urllib.request.URLError:
            attempts += 1
    if success:
        with open(param_path, 'wb') as outfile:
            outfile.write(response.read())
    else:
        raise urllib.request.URLError('Failed to download param file {}'.format(param_number))


def perform_registration_from_parameters(fixed_image,
                                         moving_image,
                                         fixed_mask=None,
                                         moving_mask=None,
                                         log_to_console=True,
                                         log_dir=None,
                                         param_file=54,
                                         param_dir='param_files',
                                         param_kwargs={},
                                         **kwargs):
    if isinstance(param_file, int):
        param_object = get_elastix_params(param_file, param_dir, overwrite=False)
    elif isinstance(param_file, sitk.ParameterMap):
        param_object = param_file
    else:
        param_object = sitk.ReadParameterFile(param_file)
    param_object['WriteResultImage'] = ('false',)
    param_object['WriteResultImageAfterEachResolution'] = ('false',)
    for key in param_kwargs:
        param_object[key] = param_kwargs[key]

    if isinstance(fixed_image, SmartImage):
        fixed_image = fixed_image.sitk_image
    if isinstance(moving_image, SmartImage):
        moving_image = moving_image.sitk_image

    elastix_filter = sitk.ElastixImageFilter()
    elastix_filter.SetParameterMap(param_object)
    elastix_filter.SetFixedImage(fixed_image)
    elastix_filter.SetMovingImage(moving_image)
    if fixed_mask is not None:
        if isinstance(fixed_mask, SmartImage):
            fixed_mask = fixed_mask.sitk_image
        elastix_filter.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        if isinstance(moving_mask, SmartImage):
            moving_mask = moving_mask.sitk_image
        elastix_filter.SetMovingMask(moving_mask)
    if log_dir is not None:
        elastix_filter.SetOutputDirectory(str(log_dir))
    elastix_filter.SetLogToConsole(log_to_console)
    result_image = elastix_filter.Execute()
    return result_image, elastix_filter.GetTransformParameterMap()[0]


def perform_transformation(moving_image, parameters, is_segmentation=False, return_deformation_field=False, log_to_console=True):
    if isinstance(parameters, str):
        parameters = sitk.ReadParameterFile(parameters)
    if is_segmentation:
        parameters['FinalBSplineInterpolationOrder'] = '0'
        parameters['DefaultPixelValue'] = '0'
    if isinstance(moving_image, SmartImage):
        moving_image = moving_image.sitk_image

    trans_filter = sitk.TransformixImageFilter()
    trans_filter.SetMovingImage(moving_image)
    trans_filter.SetTransformParameterMap(parameters)
    trans_filter.SetLogToConsole(log_to_console)
    if return_deformation_field:
        trans_filter.ComputeDeformationFieldOn()
    result = trans_filter.Execute()
    if is_segmentation:
        result = sitk.Cast(result, sitk.sitkUInt8)
    if return_deformation_field:
        def_field = trans_filter.GetDeformationField()
        return result, def_field
    return result

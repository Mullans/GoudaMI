import os

import numpy as np
import SimpleITK as sitk

from GoudaMI.optional_imports import itk


def get_image_type(image):
    """Convenience method to get object type without imports (in case itk/vtk is not installed)"""
    type_str = str(type(image))
    if 'itk.itkImage' in type_str:
        return 'itk'
    elif 'SimpleITK.ImageFileReader' in type_str:
        return 'sitkreader'
    elif 'SimpleITK.Image' in type_str:
        return 'sitk'
    elif 'NDArrayITKBase' in type_str or 'ndarray' in type_str:
        return 'numpy'
    elif 'SmartImage' in type_str:
        return 'smartimage'
    elif 'vtkPolyData' in type_str:
        return 'vtk_polydata'
    elif "<class 'dict'>" == type_str:
        return 'dict'
    elif 'goudapath' in type_str:
        return 'goudapath'
    elif isinstance(image, os.PathLike):
        return 'path'  # this should very rarely get hit
    elif "<class 'str'>" == type_str:
        return 'string'  # this could get hit for paths?
    elif "<class 'type'>" == type_str:
        return str(image)  # python types
    else:
        return type_str


class SmartType():
    # Unused types:
    # bool = -1, python int = -2, python float = -3, long double (itk) = -4
    # complex 32 = -5, complex 128 = -6
    # There are many more, but these are the ones that seem most likely to come up
    python2uniform = {bool: -1, int: -2, float: -3, complex: -6}

    numpy2uniform = {np.int8: 0, np.uint8: 1, np.int16: 2, np.uint16: 3, np.int32: 4, np.uint32: 5, np.int64: 6, np.uint64: 7, np.float32: 8, np.float64: 9, np.complex64: 11, np.complex128: -6}
    uniform2numpy = {v: k for k, v in numpy2uniform.items()}

    if hasattr(itk, 'is_dummy'):
        itk2uniform = {}
        uniform2itk = {}
    else:
        itk2uniform = {itk.B: -1, itk.SC: 0, itk.UC: 1, itk.SS: 2, itk.US: 3, itk.SI: 4, itk.UI: 5, itk.SL: 4, itk.UL: 5, itk.SLL: 6, itk.ULL: 7, itk.F: 8, itk.D: 9, itk.LD: -4}
        uniform2itk = {k: v for v, k in itk2uniform.items()}
        # NOTE - SL doesn't seem to be wrapped as often as SI, and long seems to be more platform dependent than int or long long. So we'll just default away from long types
        # https://www.intel.com/content/www/us/en/developer/articles/technical/size-of-long-integer-type-on-different-architecture-and-os.html
        uniform2itk[4] = itk.SI  # signed integer over signed long - SL often not wrapped
        uniform2itk[5] = itk.UI  # unsigned integer of unsigned long
        # ! Ignoring vector types for ITK for now... there's ~ 105 total types in ITK, and we only need 10 for numpy, so we'll just ignore the rest for now. Use itk.Image.GetTypes() to see them all. --- ITK vector types pre-define the ndim and number of components [2, 3, 4] for each type... itk.Image[itk.Vector[itk.F, 3], 3] is a 3D image with 3 components per pixel, each of type float32

    sitk2uniform = {sitk.sitkInt8: 0, sitk.sitkUInt8: 1, sitk.sitkInt16: 2, sitk.sitkUInt16: 3, sitk.sitkInt32: 4, sitk.sitkUInt32: 5, sitk.sitkInt64: 6, sitk.sitkUInt64: 7, sitk.sitkFloat32: 8, sitk.sitkFloat64: 9, sitk.sitkComplexFloat32: -5, sitk.sitkComplexFloat64: 11, sitk.sitkVectorInt8: 12, sitk.sitkVectorUInt8: 13, sitk.sitkVectorInt16: 14, sitk.sitkVectorUInt16: 15, sitk.sitkVectorInt32: 16, sitk.sitkVectorUInt32: 17, sitk.sitkVectorInt64: 18, sitk.sitkVectorUInt64: 19, sitk.sitkVectorFloat32: 20, sitk.sitkVectorFloat64: 21, sitk.sitkLabelUInt8: 22, sitk.sitkLabelUInt16: 23, sitk.sitkLabelUInt32: 24, sitk.sitkLabelUInt64: 25}

    uniform2sitk = {v: k for k, v in sitk2uniform.items()}

    # sitkstring2uniform = ', '.join(f"'{sitk.Image(1, 1, key).GetPixelIDTypeAsString()}': {val}" for key, val in SmartType.sitk2uniform.items())
    sitkstring2uniform = {
        '8-bit signed integer': 0, '8-bit unsigned integer': 1, '16-bit signed integer': 2, '16-bit unsigned integer': 3, '32-bit signed integer': 4, '32-bit unsigned integer': 5, '64-bit signed integer': 6, '64-bit unsigned integer': 7, '32-bit float': 8, '64-bit float': 9, 'complex of 32-bit float': -5, 'complex of 64-bit float': 11, 'vector of 8-bit signed integer': 12, 'vector of 8-bit unsigned integer': 13, 'vector of 16-bit signed integer': 14, 'vector of 16-bit unsigned integer': 15, 'vector of 32-bit signed integer': 16, 'vector of 32-bit unsigned integer': 17, 'vector of 64-bit signed integer': 18, 'vector of 64-bit unsigned integer': 19, 'vector of 32-bit float': 20, 'vector of 64-bit float': 21, 'label of 8-bit unsigned integer': 22, 'label of 16-bit unsigned integer': 23, 'label of 32-bit unsigned integer': 24, 'label of 64-bit unsigned integer': 25,

    }

    string2uniform = {
        'int8': 0, 'uint8': 1, 'int16': 2, 'uint16': 3, 'int32': 4, 'uint32': 5, 'int64': 6, 'uint64': 7, 'float32': 8, 'float64': 9, 'double': 9, 'complex64': 11,
        'vector int8': 12, 'vector uint8': 13, 'vector int16': 14, 'vector uint16': 15, 'vector int32': 16, 'vector uint32': 17, 'vector int64': 18, 'vector uint64': 19, 'vector float32': 20, 'vector float64': 21, 'vector double': 21,
        'label uint8': 22, 'label uint16': 23, 'label uint32': 24, 'label uint64': 25,
        'bool': -1, 'int': -2, 'float': -3, 'long double': -4, 'complex 32': -5, 'complex 128': -6
    }

    uniform2string = {v: k for k, v in string2uniform.items()}
    uniform2string[9] = 'float64'  # float64 = double, but is a better string rep.
    uniform2string[21] = 'vector float64'

    @staticmethod
    def to_uniform(data_type):
        type_name = get_image_type(data_type)
        if type_name == 'smartimage' or type_name == 'numpy':
            data_type = data_type.dtype
        elif type_name == 'sitk' or type_name == 'sitk_reader':
            data_type = data_type.GetPixelID()
        elif type_name == 'itk':
            data_type = itk.template(data_type)[1][0]
        # `else` is assumed to be a class/data type rather than a string or image object

        if isinstance(data_type, np.dtype):
            data_type = str(data_type)  # strings keys are the same as numpy string equivalents
        if 'numpy' in str(data_type):
            item_type = SmartType.numpy2uniform.get(data_type)
            if item_type is None:
                pass
            elif item_type >= 0:
                return item_type
        elif 'itkCType' in str(data_type):
            item_type = SmartType.itk2uniform.get(data_type, None)
            if item_type is None:
                pass
            elif item_type < 0:
                raise ValueError(f'ITK type `{data_type}` is not currently supported')
            elif item_type >= 0:
                return item_type
        # elif isinstance(data_type, type) and
        elif isinstance(data_type, int):
            item_type = SmartType.sitk2uniform.get(data_type)
            if item_type is not None:
                return item_type
        elif isinstance(data_type, str):
            item_type = SmartType.string2uniform.get(data_type)
            if item_type is None:
                # Backup check for SimpleITK strings, ie image.GetPixelIDTypeAsString()
                item_type = SmartType.sitkstring2uniform.get(data_type)
            if item_type is None:
                pass
            elif item_type == -1:
                raise ValueError('`bool` types are not supported for images. Please use `uint8` instead.')
            elif item_type == -2 or item_type == -3:
                raise ValueError('`int` and `float` types are not supported. Please use `int32`, `int64`, `float32`, and `float64` instead.')
            elif item_type >= 0:
                return item_type
        raise ValueError(f'Unknown data type: `{data_type}`', type(data_type))

    @staticmethod
    def as_numpy(data_type):
        data_type = SmartType.to_uniform(data_type)
        if data_type >= 12 and data_type <= 21:
            data_type = data_type - 12  # convert vector types to scalar
        return SmartType.uniform2numpy.get(data_type)

    @staticmethod
    def as_itk(data_type):
        # NOTE - probably need to handle conversion to itk vector types here... extra kwarg for num components?
        data_type = SmartType.to_uniform(data_type)
        return SmartType.uniform2itk.get(data_type)

    @staticmethod
    def as_sitk(data_type, vector=False):
        data_type = SmartType.to_uniform(data_type)
        if vector:
            if data_type < 10:
                data_type += 12
            elif data_type > 11 and data_type < 22:
                pass  # already a vector type
            elif data_type > 21:
                data_type = (data_type - 21) * 2 - 1  # convert label types to vector types
            else:
                raise ValueError('Cannot convert complex types to Vector types')
        return SmartType.uniform2sitk.get(data_type)

    @staticmethod
    def as_string(data_type):
        data_type = SmartType.to_uniform(data_type)
        return SmartType.uniform2string.get(data_type)


class SmartInterpolator():
    sitk2string = {sitk.sitkNearestNeighbor: 'nearest_neighbor',
                   sitk.sitkLinear: 'linear',
                   sitk.sitkBSpline: 'bspline'}
    string2sitk = {'nearest_neighbor': sitk.sitkNearestNeighbor,
                   'linear': sitk.sitkLinear,
                   'bspline': sitk.sitkBSpline}
    itk2string = {itk.NearestNeighborInterpolateImageFunction: 'nearest_neighbor',
                  itk.LinearInterpolateImageFunction: 'linear',
                  itk.BSplineInterpolateImageFunction: 'bspline'}
    string2itk = {'nearest_neighbor': itk.NearestNeighborInterpolateImageFunction,
                  'linear': itk.LinearInterpolateImageFunction,
                  'bspline': itk.BSplineInterpolateImageFunction}

    @staticmethod
    def to_string(interp):
        if isinstance(interp, int):
            interp_name = SmartInterpolator.sitk2string.get(interp, 'other')
            interp_type = 'sitk'
        elif isinstance(interp, itk.support.template_class.itkTemplate):
            interp_name = SmartInterpolator.itk2string.get(interp, 'other')
            interp_type = 'itk'
        elif isinstance(interp, itk.InterpolateImageFunction):
            if isinstance(interp, itk.NearestNeighborInterpolateImageFunction):
                interp_name = 'nearest_neighbor'
            elif isinstance(interp, itk.LinearInterpolateImageFunction):
                interp_name = 'linear'
            elif isinstance(interp, itk.BSplineInterpolateImageFunction):
                interp_name = 'bspline'
            else:
                interp_name = 'other'
            interp_type = 'itk'
        elif isinstance(interp, str):
            interp_name = interp
            interp_type = 'string'
        else:
            raise ValueError("Unknown interpolation type {}".format(interp))
        return interp_name, interp_type

    @staticmethod
    def as_sitk(interp):
        interp_name, interp_type = SmartInterpolator.to_string(interp)
        if interp_name == 'other':
            if interp_type != 'sitk':
                raise ValueError('Cannot understand interpolation type {} for SimpleITK image'.format(interp))
            else:
                return interp
        else:
            return SmartInterpolator.string2sitk[interp_name]

    @staticmethod
    def as_itk(interp, input_image=None):
        interp_name, interp_type = SmartInterpolator.to_string(interp)
        if interp_type != 'itk':
            if interp_name == 'other':
                raise ValueError('Cannot understand interpolation type {} for ITK image'.format(interp))
            else:
                interp = SmartInterpolator.string2itk[interp_name]
        if isinstance(interp, itk.support.template_class.itkTemplate) and input_image is not None:
            interp = interp.New(input_image)
        return interp

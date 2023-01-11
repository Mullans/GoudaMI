import importlib

import numpy as np
import SimpleITK as sitk


# UI-Lung uses (-1024, 1024), Nvidia uses (-1000, 500)
MIN_INTENSITY_PULMONARY_CT = -1000
MAX_INTENSITY_PULMONARY_CT = 500


check_itk = importlib.util.find_spec('itk')
if check_itk is None:
    DTYPE_MATCH_ITK = {}

    DTYPE_STRING = {
        'float32': [None, sitk.sitkFloat32],
        'float64': [None, sitk.sitkFloat64],
        'uint8': [None, sitk.sitkUInt8],
        'int16': [None, sitk.sitkInt16],
        'uint16': [None, sitk.sitkUInt16]
    }

else:
    import itk

    DTYPE_MATCH_ITK = {
        np.float32: itk.F,
        np.float64: itk.D,
        np.uint8: itk.UC,
        np.int16: itk.SS,
        np.uint16: itk.US,
    }

    DTYPE_STRING = {
        'float32': [itk.F, sitk.sitkFloat32],
        'float64': [itk.D, sitk.sitkFloat64],
        'uint8': [itk.UC, sitk.sitkUInt8],
        'int16': [itk.SS, sitk.sitkInt16],
        'uint16': [itk.US, sitk.sitkUInt16]
    }

DTYPE_MATCH_NP2SITK = {
    np.float32: sitk.sitkFloat32,
    np.float64: sitk.sitkFloat64,
    np.uint8: sitk.sitkUInt8,
    np.int16: sitk.sitkInt16,
    np.uint16: sitk.sitkUInt16,
}


class SmartType():
    # Unused types:
    # bool = -1, python int = -2, python float = -3, long double (itk) = -4
    # complex 32 = -5, complex 128 = -6
    # There are many more, but these are the ones that seem most likely to come up
    numpy2uniform = {np.int8: 0, np.uint8: 1, np.int16: 2, np.uint16: 3, np.int32: 4, np.uint32: 5, np.int64: 6, np.uint64: 7, np.float32: 8, np.float64: 9, np.complex64: 11, np.complex128: -6}
    uniform2numpy = {v: k for k, v in numpy2uniform.items()}

    if check_itk is None:
        itk2uniform = {}
        uniform2itk = {}
    else:
        itk2uniform = {itk.B: -1, itk.SC: 0, itk.UC: 1, itk.SS: 2, itk.US: 3, itk.SI: 4, itk.UI: 5, itk.SL: 4, itk.UL: 5, itk.SLL: 6, itk.ULL: 7, itk.F: 8, itk.D: 9, itk.LD: -4}
        uniform2itk = {k: v for v, k in itk2uniform.items()}
        # NOTE - SL doesn't seem to be wrapped as often as SI, and long seems to be more platform dependent than int or long long. So we'll just default away from long types
        # https://www.intel.com/content/www/us/en/developer/articles/technical/size-of-long-integer-type-on-different-architecture-and-os.html
        uniform2itk[4] = itk.SI  # signed integer over signed long - SL often not wrapped
        uniform2itk[5] = itk.UI  # unsigned integer of unsigned long
        #! Ignoring vector types for ITK for now... there's ~ 105 total types in ITK, and we only need 10 for numpy, so we'll just ignore the rest for now. Use itk.Image.GetTypes() to see them all.

    sitk2uniform = {sitk.sitkInt8: 0, sitk.sitkUInt8: 1, sitk.sitkInt16: 2, sitk.sitkUInt16: 3, sitk.sitkInt32: 4, sitk.sitkUInt32: 5, sitk.sitkInt64: 6, sitk.sitkUInt64: 7, sitk.sitkFloat32: 8, sitk.sitkFloat64: 9, sitk.sitkComplexFloat32: -5, sitk.sitkComplexFloat64: 11, sitk.sitkVectorInt8: 12, sitk.sitkVectorUInt8: 13, sitk.sitkVectorInt16: 14, sitk.sitkVectorUInt16: 15, sitk.sitkVectorInt32: 16, sitk.sitkVectorUInt32: 17, sitk.sitkVectorInt64: 18, sitk.sitkVectorUInt64: 19, sitk.sitkVectorFloat32: 20, sitk.sitkVectorFloat64: 21, sitk.sitkLabelUInt8: 22, sitk.sitkLabelUInt16: 23, sitk.sitkLabelUInt32: 24, sitk.sitkLabelUInt64: 25}

    uniform2sitk = {v: k for k, v in sitk2uniform.items()}

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
        elif isinstance(data_type, str):  # TODO - check for SimpleITK strings, ie image.GetPixelIDTypeAsString()
            item_type = SmartType.string2uniform.get(data_type, None)
            if item_type is None:
                pass
            elif item_type == -1:
                raise ValueError('`bool` types are not supported for images. Please use `uint8` instead.')
            elif item_type == -2 or item_type == -3:
                raise ValueError('`int` and `float` types are not supported. Please use `int32`, `int64`, `float32`, and `float64` instead.')
            elif item_type > 0:
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

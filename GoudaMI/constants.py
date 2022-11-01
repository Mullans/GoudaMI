import importlib

import numpy as np
import SimpleITK as sitk

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
    numpy2uniform = {np.int8: 0, np.uint8: 1, np.int16: 2, np.uint16: 3, np.int32: 4, np.uint32: 5, np.int64: 6, np.uint64: 7, np.float32: 8, np.float64: 9}
    uniform2numpy = {0: np.int8, 1: np.uint8, 2: np.int16, 3: np.uint16, 4: np.int32, 5: np.uint32, 6: np.int64, 7: np.uint64, 8: np.float32, 9: np.float64}

    if check_itk is None:
        itk2uniform = {}
        uniform2itk = {}
    else:
        itk2uniform = {itk.B: -1, itk.SC: 0, itk.UC: 1, itk.SS: 2, itk.US: 3, itk.SI: 4, itk.UI: 5, itk.SL: 4, itk.UL: 5, itk.SLL: 6, itk.ULL: 7, itk.F: 8, itk.D: 9, itk.LD: -1}
        uniform2itk = {0: itk.SC, 1: itk.UC, 2: itk.SS, 3: itk.US, 4: itk.SI, 5: itk.UI, 6: itk.SLL, 7: itk.ULL, 8: itk.F, 9: itk.D}

    sitk2uniform = {sitk.sitkInt8: 0, sitk.sitkUInt8: 1, sitk.sitkInt16: 2, sitk.sitkUInt16: 3, sitk.sitkInt32: 4, sitk.sitkUInt32: 5, sitk.sitkInt64: 6, sitk.sitkUInt64: 7, sitk.sitkFloat32: 8, sitk.sitkFloat64: 9}
    uniform2sitk = {0: sitk.sitkInt8, 1: sitk.sitkUInt8, 2: sitk.sitkInt16, 3: sitk.sitkUInt16, 4: sitk.sitkInt32, 5: sitk.sitkUInt32, 6: sitk.sitkInt64, 7: sitk.sitkUInt64, 8: sitk.sitkFloat32, 9: sitk.sitkFloat64}

    string2uniform = {'int8': 0, 'uint8': 1, 'int16': 2, 'uint16': 3, 'int32': 4, 'uint32': 5, 'int64': 6, 'uint64': 7, 'float32': 8, 'float64': 9, 'double': 9, 'int': -1, 'float': -1}
    uniform2string = {0: 'int8', 1: 'uint8', 2: 'int16', 3: 'uint16', 4: 'int32', 5: 'uint32', 6: 'int64', 7: 'uint64', 8: 'float32', 9: 'float64'}

    @staticmethod
    def to_uniform(data_type):
        if 'numpy' in str(data_type) or isinstance(data_type, np.dtype):
            item_type = SmartType.numpy2uniform.get(data_type, -2)
            if item_type != -2:
                return item_type
        elif 'itkCType' in str(data_type):
            item_type = SmartType.itk2uniform.get(data_type, -2)
            if item_type == -1:
                raise ValueError(f'ITK type `{data_type}` is not currently supported')
            elif item_type != -2:
                return item_type
        elif isinstance(data_type, int):
            item_type = SmartType.sitk2uniform.get(data_type, -2)
            if item_type != -2:
                return item_type
        elif isinstance(data_type, str):
            item_type = SmartType.string2uniform.get(data_type, -2)
            if item_type == -1:
                raise ValueError(f'`int` and `float` types are not supported. Please use `int32`, `int64`, `float32`, and `float64` instead.')
            elif item_type != -2:
                return item_type
        raise ValueError(f'Unknown data type: `{data_type}`')

    @staticmethod
    def as_numpy(data_type):
        data_type = SmartType.to_uniform(data_type)
        return SmartType.uniform2numpy.get(data_type)

    @staticmethod
    def as_itk(data_type):
        data_type = SmartType.to_uniform(data_type)
        return SmartType.uniform2itk.get(data_type)

    @staticmethod
    def as_sitk(data_type):
        data_type = SmartType.to_uniform(data_type)
        return SmartType.uniform2sitk.get(data_type)

    @staticmethod
    def as_string(data_type):
        data_type = SmartType.to_uniform(data_type)
        return SmartType.uniform2string.get(data_type)
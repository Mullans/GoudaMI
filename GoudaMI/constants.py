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

DTYPE_MATCH_SITK = {
    np.float32: sitk.sitkFloat32,
    np.float64: sitk.sitkFloat64,
    np.uint8: sitk.sitkUInt8,
    np.int16: sitk.sitkInt16,
    np.uint16: sitk.sitkUInt16,
}

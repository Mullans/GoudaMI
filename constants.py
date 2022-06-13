import itk
import numpy as np
import SimpleITK as sitk


DTYPE_MATCH_ITK = {
    np.float32: itk.F,
    np.float64: itk.D,
    np.uint8: itk.UC,
    np.int16: itk.SS,
    np.uint16: itk.US,
}


DTYPE_MATCH_SITK = {
    np.float32: sitk.sitkFloat32,
    np.float64: sitk.sitkFloat64,
    np.uint8: sitk.sitkUInt8,
    np.int16: sitk.sitkInt16,
    np.uint16: sitk.sitkUInt16,
}

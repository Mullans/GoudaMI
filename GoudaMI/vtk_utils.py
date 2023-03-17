from GoudaMI.convert import wrap4vtk
from GoudaMI.optional_imports import vtk


def vtk_copy_information(image, ref):
    if hasattr(image, 'SetDimensions'):
        image.SetDimensions(ref.GetDimensions())
    if hasattr(image, 'SetOrigin'):
        image.SetOrigin(ref.GetOrigin())
        image.SetSpacing(ref.GetSpacing())
        image.SetExtent(ref.GetExtent())
    elif hasattr(image, 'SetOutputOrigin'):
        image.SetOutputOrigin(ref.GetOrigin())
        image.SetOutputSpacing(ref.GetSpacing())
        image.SetOutputWholeExtent(ref.GetExtent())
    return image


def get_sampling_info(item):
    info = {
        'origin': item.GetOrigin(),
        'spacing': item.GetSpacing(),
        'extent': item.GetExtent()
    }
    return info


@wrap4vtk
def get_smoothed_contour(contour, num_iterations=20, pass_band=0.01):
    """Apply the vtkWindowedSincPolyData Filter to a vtkImageData

    Parameters
    ----------
    contour : vtk.ImageData
        The contour to be smoothed
    num_iterations : int
        Number of iterations for the vtkWindowedSincPolyDataFilter
    pass_band : float
        The pass band for vtkWindowedSincPolyDataFilter

    Returns
    -------
    vtk.PolyData
    """
    discrete = vtk.vtkDiscreteFlyingEdges3D()
    discrete.SetInputData(contour)
    # discrete.GenerateValues(n, 1, n)
    discrete.GenerateValues(1, 1, 1)
    discrete.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(discrete.GetOutput())
    smoother.SetNumberOfIterations(num_iterations)
    smoother.SetPassBand(pass_band)
    smoother.Update()

    return smoother.GetOutput()

import glob
import os
import warnings

import numpy as np

from .smart_image import SmartImage, get_image_type


# def write_vtk(data, output_path, output_version=42):
#     import vtk
#     writer = vtk.vtkDataWriter()
#     writer.SetFileVersion(output_version)
#     writer.SetInputData(data)
#     writer.SetFileName(output_path)
#     writer.Update()
#     return writer
def write_vtk(polydata, filename, is_label=True):
    import vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(filename)
    if is_label:
        writer.SetFileTypeToBinary()
    writer.Update()


def read_vtk(path, data_type='polydata'):
    import vtk
    if data_type == 'polydata':
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError("Unknown data type {}. Add it to the method if it's correct".format(data_type))
    reader.SetInputString(path)
    reader.Update()
    return reader.GetOutput()


def read_sitk(path, allow_search=True):
    import gouda
    import SimpleITK as sitk
    """Read a dicom image in either directory or single file formats"""
    if isinstance(path, gouda.GoudaPath):
        path = path.path
    if os.path.isdir(path):
        dicom_files = sorted(glob.glob(os.path.join(path, '*.dcm')))
        if len(dicom_files) == 0 and allow_search:
            possible = search_for_dicom(path)
            if len(possible) == 0:
                raise ValueError('No dicom images found within "{}"'.format(path))
            elif len(possible) > 1:
                warnings.warn('Multiple dicom directories found. Using "{}"'.format(possible[0]))
            else:
                pass
            path = possible[0]
            dicom_files = sorted(glob.glob(os.path.join(path, '*.dcm')))

        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(dicom_files[0])
        file_reader.ReadImageInformation()
        series_id = file_reader.GetMetaData('0020|000e')
        sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_id)
        return sitk.ReadImage(sorted_file_names)
    else:
        return sitk.ReadImage(path)


def read_dicom_as_sitk(path):
    import SimpleITK as sitk
    """Similar to read_sitk, but dicom specific and includes all meta tags"""
    possible = search_for_dicom(path)
    if len(possible) == 0:
        raise ValueError('No dicom images found within "{}"'.format(path))
    elif len(possible) > 1:
        warnings.warn('Multiple dicom directories found. Using "{}"'.format(possible[0]))
    else:
        pass
    path = possible[0]

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    if not series_IDs:
        raise ValueError('Directory {} does not contain dicoms'.format(path))
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    series_reader.MetaDataDictionaryArrayUpdateOn()  # needed to load tags
    series_reader.LoadPrivateTagsOn()  # needed to load private tags too
    image = series_reader.Execute()

    return image, series_reader


def search_for_dicom(base_dir):
    possible = []
    for root, dirs, files in os.walk(base_dir):
        has_dicom = False
        for item in files:
            if item.endswith('.dcm'):
                has_dicom = True
                break
        if has_dicom:
            possible.append(root)
    return possible


def write_image(data, path, label=None, skip_existing_filenames=False):
    """Write an image to disk

    Parameters
    ----------
    data: SimpleITK.Image or itk.Image or numpy.ndarray
        The data to save
    path : str
    """
    image_type = get_image_type(data)
    if image_type == 'sitk':
        import SimpleITK as sitk
    # if isinstance(data, sitk.Image):
        if path.endswith('.npy'):
            data = sitk.GetArrayViewFromImage(data)
            np.save(path, data)
        else:
            sitk.WriteImage(data, path)
    elif image_type == 'itk':
    # elif isinstance(data, itk.Image):
        import itk
        if path.endswith('.npy'):
            data = itk.GetArrayViewFromImage(data)
            np.save(path, data)
        else:
            itk.imwrite(data, path)
    elif isinstance(data, np.ndarray):
        if path.endswith('.npy'):
            np.save(path, data)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            import SimpleITK as sitk
            data = sitk.GetImageFromArray(data)
            sitk.WriteImage(data, path)
    else:
        raise ValueError('Unknown data type: {}'.format(type(data)))


def parse_color(color, float_cmap='viridis', int_cmap='Set1'):
    """Convert the input to a rgb color

    NOTE
    ----
    Recognizes all formats that can be used with matplotlib in addition to rgb/rgba tuples as strings [ie. '(0.1, 0.2, 0.5)'] and single values.
    Single values will be mapped to the given matplotlib colormap. Ints will wrap-around and floats will be clipped to [0, 1].

    TODO: DELETE ME WHEN ADDED TO GOUDA LIVE VERSION
    """
    import matplotlib
    try:
        return matplotlib.colors.to_rgb(color)
    except ValueError:
        if isinstance(color, str):
            # Format is comma- and/or space-separated values
            color.translate(None, '()[]')
            if ', ' in color:
                divided = color.split(', ')
            elif ',' in color:
                divided = color.split(',')
            else:
                divided = color.split(' ')
            rgb = np.array(divided).astype(np.float32)
            return matplotlib.colors.to_rgb(rgb / 255 if rgb.max() > 1.0 else rgb)
        elif isinstance(color, float):
            return matplotlib.cm.get_cmap(float_cmap)
        elif isinstance(color, int):
            return matplotlib.cm.get_cmap(int_cmap)(color % 9)
        else:
            # Format is any array-like set of values
            rgb = np.array(color).astype(np.float32)
            return matplotlib.colors.to_rgb(rgb / 255 if rgb.max() > 1.0 else rgb)


CT_LABEL_COLORS = ['red', 'deepskyblue', 'lime', 'fuchsia', 'darkorange', 'yellow', 'chocolate']


def write_sitk_seg_for_slicer(data, path, colors=None, segment_labels=None, segment_prefix=None, background_val=0, compression=10):
    """Write a SimpleITK label map as an nrrd segmentation for 3DSlicer.

    NOTE
    ----
    Does not work with overlapping labels (data has to be a label image with single labels per voxel)

    """
    from .ct_utils import get_label_bounds
    import SimpleITK as sitk

    if isinstance(data, SmartImage):
        data = data.sitk_image

    label_bounds = get_label_bounds(data, background_val=background_val)
    num_segments = len(label_bounds)

    if segment_labels is None:
        segment_labels = ['Segment{}'.format(idx) for idx in range(num_segments)]

    if colors is None:
        colors = [i for i in range(num_segments)]
    if isinstance(colors, str) or not hasattr(colors, '__iter__'):
        colors = [colors]
    colors = [parse_color(color) for color in colors]
    colors = ['{:.6f} {:.6f} {:.6f}'.format(*color) for color in colors]

    for idx, label_num in enumerate(label_bounds.keys()):
        if segment_prefix is not None:
            data.SetMetaData('Segment{}_ID'.format(idx), '{}_{}'.format(segment_prefix, segment_labels[idx]))
        else:
            data.SetMetaData('Segment{}_ID'.format(idx), segment_labels[idx])
        data.SetMetaData('Segment{}_Name'.format(idx), segment_labels[idx])
        data.SetMetaData('Segment{}_NameAutoGenerated'.format(idx), '0')
        data.SetMetaData('Segment{}_Color'.format(idx), colors[idx % len(colors)])
        data.SetMetaData('Segment{}_ColorAutoGenerated'.format(idx), '0')
        data.SetMetaData('Segment{}_Layer'.format(idx), '0')
        data.SetMetaData('Segment{}_LabelValue'.format(idx), '{:d}'.format(label_num))
    data.SetMetaData('Segmentation_MasterRepresentation', 'Binary labelmap')
    data.SetMetaData('Segmentation_ContainedRepresentationNames', 'Binary labelmap')
    # sitk.WriteImage(data, path, useCompression=compression > 0, compressionLevel=compression)
    ## Swig doesn't allow keyword arguments in overloaded functios?
    sitk.WriteImage(data, str(path), compression > 0, compression)

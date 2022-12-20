import glob
# import importlib
import os
import warnings
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from GoudaMI.constants import SmartType

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
    else:
        return None


def get_dtype(image):
    """Convenience method to get object pixel-type without imports"""
    image_type = get_image_type(image)
    if image_type == 'itk':
        out_type = itk.template(image)[1][0]
    elif image_type == 'sitk':
        out_type = image.GetPixelID()
    elif image_type == 'numpy' or image_type == 'smartimage':
        out_type = image.dtype
    elif image_type == 'dict':
        out_type = image['dtype']
    else:
        return None
    return SmartType.as_string(out_type)


def _search_for_dicom(base_dir):
    # TODO - replicated from io to prevent cycle - add to GOUDA
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


def sitk2itk(image):
    """Local version of the method in convert to avoid circular imports"""
    if isinstance(image, itk.Image):
        return image
    itk_image = itk.GetImageFromArray(sitk.GetArrayViewFromImage(image), is_vector=image.GetNumberOfComponentsPerPixel() > 1)
    itk_image.SetOrigin(image.GetOrigin())
    itk_image.SetSpacing(image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [image.GetDimension()] * 2)))
    return itk_image


def itk2sitk(image):
    """Local version of the method in convert to avoid circular imports"""
    if isinstance(image, sitk.Image):
        return image
    sitk_image = sitk.GetImageFromArray(itk.GetArrayViewFromImage(image), isVector=image.GetNumberOfComponentsPerPixel() > 1)
    sitk_image.SetOrigin(tuple(image.GetOrigin()))
    sitk_image.SetSpacing(tuple(image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return sitk_image


class SmartImage(object):
    def __init__(self, path, default_type='sitk'):
        self.__sitk_image: sitk.Image = None
        self.__itk_image: itk.Image = None
        self.__updated_itk: bool = False
        self.__updated_sitk: bool = False
        self.__image_type: str = None
        self.__meta_data: dict = None
        if default_type not in ['sitk', 'itk']:
            raise ValueError('SmartImage default type must be `sitk` or `itk`.')
        self.default_type = default_type
        # These are calculated only as needed
        self.__reset_internals()

        if hasattr(path, 'abspath'):
            path = path.abspath
        if isinstance(path, sitk.ImageFileReader):
            path = path.GetFileName()
            if path == '':
                raise ValueError('Cannot initialize SmartImage with empty file reader')
        if not isinstance(path, (str, os.PathLike)):
            input_type = get_image_type(path)
            if input_type == 'itk':
                self.__itk_image = path
                self.__sitk_image = None
                self.__updated_itk = True
                self.path = ""
            elif input_type == 'sitk':
                self.__sitk_image = path
                self.__itk_image = None
                self.__updated_sitk = True
                self.path = ""
            elif input_type == 'smartimage':
                # NOTE: This is just a shallow copy of the other SmartImage for convenience
                self.default_type = path.default_type
                self.__sitk_image = path.__sitk_image
                self.__itk_image = path.__itk_image
                self.__updated_itk = path.__updated_itk
                self.__updated_sitk = path.__updated_sitk
                self.path = path.path
            elif input_type == 'numpy':
                if path.ndim != 3:
                    raise ValueError("SmartImage can only be initialized with volumes for now")
                if path.dtype == bool:
                    path = path.astype(np.uint8)
                self.__sitk_image = sitk.GetImageFromArray(path)
                self.__itk_image = None
                self.__updated_sitk = True
                self.path = ''
            else:
                raise ValueError("Unsupported datatype: {}".format(type(path)))
        else:
            self.path = path
        # self.image

    def __reset_internals(self, direction_only=False):
        if direction_only:
            # NOTE - only reset values where direction matters. None currently
            return
        self.__minimum_intensity = None
        self.__maximum_intensity = None
        self.__unique_labels = None
        self.__mean_intensity = None
        self.__sum_intensity = None
        self.__stddev_intensity = None
        self.__var_intensity = None

    def __run_minmax(self):
        image = self.image
        if isinstance(image, sitk.Image):
            filt = sitk.MinimumMaximumImageFilter()
            filt.Execute(image)
            self.__maximum_intensity = filt.GetMaximum()
            self.__minimum_intensity = filt.GetMinimum()
        elif isinstance(image, itk.Image):
            raise NotImplementedError('This is on the todo list')

    def __run_image_stats(self):
        image = self.image
        if self.image_type == 'sitk':
            filt = sitk.StatisticsImageFilter()
            filt.Execute(image)
            self.__maximum_intensity = filt.GetMaximum()
            self.__minimum_intensity = filt.GetMinimum()
            self.__mean_intensity = filt.GetMean()
            self.__sum_intensity = filt.GetSum()
            self.__stddev_intensity = filt.GetSigma()
            self.__var_intensity = filt.GetVariance()
        elif self.image_type == 'itk':
            raise NotImplementedError('ITK backed stats are yet-to-come')

    def min(self):
        if self.__minimum_intensity is None:
            self.__run_minmax()
        return self.__minimum_intensity

    def max(self):
        if self.__maximum_intensity is None:
            self.__run_minmax()
        return self.__maximum_intensity

    def mean(self):
        if self.__mean_intensity is None:
            self.__run_image_stats()
        return self.__mean_intensity

    def sum(self):
        if self.__sum_intensity is None:
            self.__run_image_stats()
        return self.__sum_intensity

    def volume(self):
        if self.__sum_intensity is None:
            self.__run_image_stats()
        pixel_volume = np.prod(self.GetSpacing())
        return self.sum() * pixel_volume

    def stddev(self):
        if self.__stddev_intensity is None:
            self.__run_image_stats()
        return self.__stddev_intensity

    def variance(self):
        if self.__var_intensity is None:
            self.__run_image_stats()
        return self.__var_intensity

    @property
    def dtype(self):
        if not self.loaded:
            dtype = self.__load_meta('dtype')
            if dtype is not None:
                return dtype
        image = self.image
        if self.image_type == 'itk':
            return SmartType.as_string(itk.template(image)[1][0])
        elif self.image_type == 'sitk':
            return SmartType.as_string(image.GetPixelID())

    @property
    def image_type(self):
        if self.__image_type is None:
            if self.__itk_image is None and self.__sitk_image is not None:
                return 'sitk'
            elif self.__itk_image is not None and self.__sitk_image is None:
                return 'itk'
            else:
                # Should it return something else if neither is loaded?
                return self.default_type
        return self.__image_type

    @property
    def image(self):
        if self.__sitk_image is None and self.__itk_image is None:
            # neither type is loaded
            if self.default_type == 'sitk':
                return_img = self.sitk_image
            elif self.default_type == 'itk':
                return_img = self.itk_image
            else:
                raise ValueError("Unknown default image type: {}".format(self.default_type))
        elif (self.__sitk_image is None or self.__updated_itk) != (self.__itk_image is None or self.__updated_sitk):
            # return the most updated and loaded type
            return_img = self.__sitk_image if self.__sitk_image is not None else self.__itk_image
        elif self.default_type == 'sitk':
            return_img = self.sitk_image
        elif self.default_type == 'itk':
            return_img = self.itk_image
        else:
            raise ValueError("No possible images to return...")

        self.__image_type = get_image_type(return_img)
        return return_img

    @property
    def itk_image(self):
        if self.__updated_sitk:
            self.__reset_internals()
            self.__itk_image = sitk2itk(self.sitk_image)
            self.__updated_sitk = False
        if self.__itk_image is None:
            self.__itk_image = itk.imread(str(self.path))
        return self.__itk_image

    @property
    def sitk_image(self):
        if self.__updated_itk:
            self.__reset_internals()
            self.__sitk_image = itk2sitk(self.itk_image)
            self.__updated_itk = False
        if self.__sitk_image is None:
            if os.path.isdir(self.path):
                dicom_files = sorted(glob.glob(os.path.join(self.path, '*.dcm')))
                if len(dicom_files) == 0:
                    possible = _search_for_dicom(self.path)
                    if len(possible) == 0:
                        raise ValueError('No dicom images found within "{}"'.format(self.path))
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
                sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(self.path, series_id)
                self.__sitk_image = sitk.ReadImage(sorted_file_names)
            else:
                self.__sitk_image = sitk.ReadImage(self.path)
        return self.__sitk_image

    @property
    def loaded(self):
        return not (self.__sitk_image is None and self.__itk_image is None)

    @property
    def ndim(self):
        if not self.loaded:
            ndim = self.__load_meta('ndim')
            if ndim is not None:
                return ndim
        image = self.image
        if self.image_type == 'sitk':
            return image.GetDimension()
        elif self.image_type == 'itk':
            return image.GetImageDimension()

    def as_array(self):
        image = self.image
        if self.image_type == 'itk':
            return itk.GetArrayFromImage(image)
        elif self.image_type == 'sitk':
            return sitk.GetArrayFromImage(image)
        else:
            raise ValueError("This should never reach here")

    def as_view(self):
        image = self.image
        if self.image_type == 'itk':
            return itk.GetArrayViewFromImage(image)
        elif self.image_type == 'sitk':
            return sitk.GetArrayViewFromImage(image)
        else:
            raise ValueError("This should never reach here")

    def astype(self, dtype, in_place=False, image_type=None, return_smart_image=True):
        image_type = self.default_type if image_type is None else image_type
        if image_type == 'sitk':
            image = self.sitk_image
            sitk_type = SmartType.as_sitk(dtype)
            if sitk_type is None:
                raise ValueError(f"Unknown dtype: {dtype}")
            image = sitk.Cast(image, sitk_type)
            if in_place:
                return self.update(image)
        elif image_type == 'itk':
            image = self.itk_image
            itk_dtype = SmartType.as_itk(dtype)
            if itk_dtype is None:
                raise ValueError(f'Unknown dtype: {dtype}')
            dim = image.GetImageDimension()
            image = itk.CastImageFilter[image, itk.Image[itk_dtype, dim]].New(image)
            if in_place:
                return self.update(image)
        else:
            raise ValueError('Unknown image type: {}'.format(image_type))
        if return_smart_image:
            return SmartImage(image)
        return image

    def __load_meta(self, key):
        if not isinstance(self.path, (os.PathLike, str)) or self.path == '':
            warnings.warn('Failed to load image header - please set path to correct file location')
            return None
        if self.__meta_data is None:
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(self.path)
            file_reader.ReadImageInformation()
            self.__meta_data = {
                'size': np.array(file_reader.GetSize()),
                'origin': np.array(file_reader.GetOrigin()),
                'spacing': np.array(file_reader.GetSpacing()),
                'direction': np.array(file_reader.GetDirection()),
                'dtype': SmartType.as_string(file_reader.GetPixelID()),
                'ndim': file_reader.GetDimension()
            }
        return self.__meta_data[key]

    def GetDirection(self):
        if not self.loaded:
            direction = self.__load_meta('direction')
            if direction is not None:
                return direction
        image = self.image
        if self.image_type == 'sitk':
            return np.array(image.GetDirection())
        elif self.image_type == 'itk':
            return itk.GetArrayFromMatrix(image.GetDirection()).flatten()

    def TransformIndexToPhysicalPoint(self, index):
        if not self.loaded:
            shift = (self.GetDirectionMatrix() * (self.GetSpacing() * np.eye(self.ndim))) @ np.asarray(index)
            return self.GetOrigin() + shift
        image = self.image
        if self.image_type == 'sitk':
            return np.array(image.TransformIndexToPhysicalPoint(index))
        elif self.image_type == 'itk':
            return np.array(image.TransformIndexToPhysicalPoint(index))

    def SetDirection(self, direction):
        self.image  # load the image if necessary
        if self.image_type == 'sitk':
            if isinstance(direction, (tuple, list)) or (isinstance(direction, np.ndarray) and direction.ndim == 1):
                pass  # assume from another sitk.Image or SmartImage
            elif isinstance(direction, np.ndarray):
                direction = direction.flatten()
            elif isinstance(direction, itk.Matrix):
                direction = itk.GetArrayFromMatrix(direction).flatten()
            else:
                raise TypeError('Unknown direction type: {}'.format(type(direction)))
            self.__sitk_image.SetDirection(direction)
            self.__updated_sitk = True
        elif self.image_type == 'itk':
            if isinstance(direction, (tuple, list)) or (isinstance(direction, np.ndarray) and direction.ndim == 1):
                direction = np.reshape(np.array(direction), [self.ndim] * 2)
                direction = itk.GetMatrixFromArray(direction)
            elif isinstance(direction, itk.Matrix):
                pass
            elif isinstance(direction, np.ndarray):
                direction = np.squeeze(direction)
                if direction.ndim != 2:
                    raise ValueError('Direction matrices must be 1- or 2d.')
                direction = itk.GetMatrixFromArray(direction)
            else:
                raise TypeError('Unknown direction type: {}'.format(type(direction)))
            self.__itk_image.SetDirection(direction)
            self.__updated_itk = True
        self.__reset_internals(direction_only=True)

    def GetDirectionMatrix(self):
        image = self.image
        if not self.loaded:
            direction = self.__load_meta('direction')
            if direction is not None:
                direction = np.array(direction)
                nrows = int(np.sqrt(direction.shape[0]))
                return direction.reshape([nrows, -1])
        if self.image_type == 'sitk':
            direction = np.array(image.GetDirection())
            nrows = int(np.sqrt(direction.shape[0]))
            return direction.reshape([nrows, -1])
        elif self.image_type == 'itk':
            return itk.GetArrayFromMatrix(image.GetDirection())

    def get_physical_properties(self):
        return {
            'size': self.GetSize(),
            'spacing': self.GetSpacing(),
            'origin': self.GetOrigin(),
            'direction': self.GetDirection(),
            'dtype': self.dtype
        }

    def GetOrigin(self):
        if not self.loaded:
            origin = self.__load_meta('origin')
            if origin is not None:
                return origin
        return np.array(self.image.GetOrigin())

    def GetOppositeCorner(self):
        """Return the coordinates of the corner opposite to the origin"""
        # TODO - double check that this is correct, Slicer seems to disagree
        size = self.GetPhysicalSize()
        # TODO - update if needed, this assumes only 180 flipped directions
        direction = self.GetDirectionMatrix()
        direction = np.array([direction[i, i] for i in range(direction.shape[0])])
        return self.GetOrigin() + size * direction

    def GetSize(self):
        if not self.loaded:
            size = self.__load_meta('size')
            if size is not None:
                return size
        image = self.image
        if self.image_type == 'sitk':
            return np.array(image.GetSize())
        elif self.image_type == 'itk':
            return np.array(image.GetLargestPossibleRegion().GetSize())

    def GetPhysicalSize(self):
        return self.GetSize() * self.GetSpacing()

    def GetSpacing(self):
        if not self.loaded:
            spacing = self.__load_meta('spacing')
            if spacing is not None:
                return spacing
        return np.array(self.image.GetSpacing())

    def CopyInformation(self, ref_image):
        image = self.image
        if isinstance(ref_image, sitk.Image) and isinstance(image, sitk.Image):
            # Easiest case: sitk -> sitk
            image.CopyInformation(ref_image)
        elif isinstance(ref_image, SmartImage) and isinstance(ref_image.image, sitk.Image) and isinstance(image, sitk.Image):
            image.CopyInformation(ref_image.image)
        elif isinstance(ref_image, dict):
            image.SetOrigin(ref_image['origin'])
            image.SetDirection(ref_image['direction'])
            image.SetSpacing(ref_image['spacing'])
        else:
            if not isinstance(ref_image, SmartImage):
                ref_image = SmartImage(ref_image)
            image.SetOrigin(ref_image.GetOrigin())
            image.SetDirection(ref_image.GetDirection())
            image.SetSpacing(ref_image.GetSpacing())

        if self.image_type == 'sitk':
            self.__updated_sitk = True
        elif self.image_type == 'itk':
            self.__updated_itk = True
        else:
            raise ValueError('Should never be here')
        return self

    # TODO - allow setting each physical value

    def label_shape_stats(self, bg_val: float = 0, perimeter: bool = False, feret_diameter: bool = False, oriented_bounding_box: bool = False) -> sitk.LabelShapeStatisticsImageFilter:
        """Create, execute, and return a :class:`SimpleITK.LabelShapeStatisticsImageFilter`

        Parameters
        ----------
        bg_val : float, optional
            The background value of the label image, by default 0
        perimeter : bool, optional
            Whether to compute the label perimeters, by default False
        feret_diameter : bool, optional
            Whether to compute the maximum Feret diameter for labels, by default False
        oriented_bounding_box : bool, optional
            Whether to compute the oriented bounding box for labels, by default False

        Returns
        -------
        SimpleITK.LabelShapeStatisticsImageFilter
            A SimpleITK filter that has already calculated the label shape statistics
        """
        if 'int' not in self.dtype:
            raise TypeError(f'Integer pixel type required for LabelShapeStatistics filter (current: {self.dtype})')
        label_filt = sitk.LabelShapeStatisticsImageFilter()
        label_filt.SetBackgroundValue(bg_val)
        label_filt.SetComputePerimeter(perimeter)
        label_filt.SetComputeFeretDiameter(feret_diameter)
        label_filt.SetComputeOrientedBoundingBox(oriented_bounding_box)
        label_filt.Execute(self.sitk_image)
        self.__unique_labels = label_filt.GetLabels()
        return label_filt

    @property
    def cc(self) -> sitk.ConnectedComponentImageFilter:
        """Short-hand for connected components so we can do ``image.cc.GetNumberOfObjects()`` or similar"""
        return self.connected_components(fully_connected=False, in_place=False)

    def connected_components(self, fully_connected=False, in_place=False):
        # TODO - add itk version, may need internal method to wrap filter
        return self.__perform_op(sitk.ConnectedComponent, None, fully_connected, in_place=in_place)

    def unique(self):
        if self.__unique_labels is None:
            label_filt = sitk.LabelShapeStatisticsImageFilter()
            if self.__minimum_intensity is not None:
                label_filt.SetBackgroundValue(self.__minimum_intensity - 1)
            else:
                label_filt.SetBackgroundValue(-1)  # -1000 makes it lose labels sometimes?
            label_filt.SetComputePerimeter(False)
            label_filt.Execute(self.sitk_image)
            self.__unique_labels = label_filt.GetLabels()
        return self.__unique_labels

    def update(self, image):
        image_type = get_image_type(image)
        if image_type == 'sitk':
            self.__sitk_image = image
            self.__updated_sitk = True
        elif image_type == 'itk':
            self.__itk_image = image
            self.__updated_itk = True
        elif image_type == 'smartimage':
            return self.update(image.image)
        else:
            raise TypeError('SmartImage must be updated with itk.Image, SimpleITK.Image, or SmartImage objects')
        self.__reset_internals()
        return self

    def sitk_op(self, op, *args, in_place=True, **kwargs):
        result = op(self.sitk_image, *args, **kwargs)
        if in_place:
            return self.update(result)
        else:
            return SmartImage(result)

    def itk_op(self, op, *args, in_place=True, **kwargs):
        result = op(self.itk_image, *args, **kwargs)
        if in_place:
            return self.update(result)
        else:
            return SmartImage(result)

    def write(self, dest_path, image_type=None, compression=0):
        image_type = self.default_type if image_type is None else image_type
        dest_path = str(dest_path)
        if image_type == 'sitk':
            image = self.sitk_image
            if compression > 0:
                sitk.WriteImage(image, dest_path, True, compression)
            else:
                sitk.WriteImage(image, dest_path)
        elif image_type == 'itk':
            image = self.itk_image
            itk.imwrite(image, dest_path)
        elif image_type == 'numpy':
            image = self.as_array()
            np.save(dest_path, image)

    def window(self, min=None, max=None, level=None, width=None, output_min=None, output_max=None, in_place=False):
        """Clip the intensity values of the image using either min/max or level/width

        Parameters
        ----------
        min : int or float, optional
            Minimum value of the window, by default None
        max : int or float, optional
            Maximum value of the window, by default None
        level : int or float, optional
            Center of the window, by default None
        width : int or float, optional
            Width of the window, by default None
        output_min: int, optional
            Minimum value of the output image, by default the same as the clipping window
        output_max: int, optional
            Maximum value of the output image, by default the same as the clipping window
        in_place : bool, optional
            Whether to update the image object or return a new image, by default False

        Returns
        -------
        SmartImage
            The result image with windowed intensity values
        """
        if min is not None and max is not None:
            pass
        elif level is not None and width is not None:
            min = level - (width / 2)
            max = level + (width / 2)
        else:
            raise ValueError('Either min/max or level/width must be set for windowing')
        output_min = min if output_min is None else output_min
        output_max = max if output_max is None else output_max
        image = self.image
        if self.image_type == 'sitk':
            result = sitk.IntensityWindowing(image, windowMinimum=min, windowMaximum=max, outputMinimum=output_min, outputMaximum=output_max)
        elif self.image_type == 'itk':
            filter = itk.IntensityWindowingImageFilter.New(image)
            filter.SetInput(image)
            filter.SetOutputMaximum(output_max)
            filter.SetOutputMinimum(output_min)
            filter.SetWindowMaximum(max)
            filter.SetWindowMinimum(min)
            filter.Update()
            result = filter.GetOutput()

        if in_place:
            self.update(result)
            return result
        return SmartImage(result)

    def resample(self,
                 size: Union[int, List[int], None, np.ndarray] = None,
                 origin: Union[List[float], None, np.ndarray] = None,
                 spacing: Union[float, List[float], None, np.ndarray] = None,
                 direction: Union[List[float], None, np.ndarray] = None,
                 outside_val: float = -1000,
                 interp: int = sitk.sitkBSpline,
                 presmooth: Optional[Union[bool, float]] = None,
                 dryrun: bool = False,
                 in_place: bool = False):
        """Resample the image to the given parameters

        Parameters
        ----------
        size : int | list | None | np.ndarray
            The output size of the image - will default to the input size if None
        origin : list | None | np.ndarray
            The output origin of the image - will default to the input origin if None
        spacing : float | list | None | np.ndarray
            The output spacing of the image - will default to the input spacing if None
        direction : list | None | np.ndarray
            The output direction of the image - will default to the input direction if None
        outside_val : float
            The default pixel value to use - the default is -1000
        interp : int
            The interpolation method to use - the default is SimpleITK.sitkBSpline
        presmooth : Optional[Union[bool, float]]
            Whether to apply a gaussian smoothing before resampling - if None, will use a sigma of 1 if the output size is smaller than the input size in all dimensions - if float, will use that as the smoothing sigma, by default None
        dryrun : bool
            If true, returns the result physical properties without performing the resampling
        in_place : bool
            If true, modifies the current image. Otherwise, returns the resampled copy.
        """
        image = self.sitk_image
        origin = self.GetOrigin() if origin is None else origin
        direction = self.GetDirection() if direction is None else direction

        if size is None and spacing is not None:
            if not hasattr(spacing, '__iter__'):
                spacing = [spacing] * self.ndim
            size = (self.GetSize() * self.GetSpacing()) / np.array(spacing)
        elif spacing is None:
            size = self.GetSize() if size is None else size
            if not hasattr(size, '__iter__'):
                size = [size] * self.ndim
            spacing = (self.GetSize() * self.GetSpacing()) / np.array(size)
        else:
            if not hasattr(size, '__iter__'):
                size = [size] * self.ndim
            if not hasattr(spacing, '__iter__'):
                spacing = [spacing] * self.ndim

        if dryrun:
            return {'size': size, 'origin': origin, 'spacing': spacing, 'direction': direction}

        if presmooth is None:
            presmooth = np.all(self.GetSize() > size)
        presmooth = float(presmooth)
        size = np.array(size).tolist()
        spacing = np.array(spacing).tolist()
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interp)
        resample_filter.SetDefaultPixelValue(outside_val)
        resample_filter.SetSize(size)
        resample_filter.SetOutputOrigin(origin)
        resample_filter.SetOutputSpacing(spacing)
        resample_filter.SetOutputDirection(direction)
        if presmooth > 0:
            image = sitk.RecursiveGaussian(image, presmooth)
        result = resample_filter.Execute(image)
        if in_place:
            self.update(result)
            return self
        else:
            return SmartImage(result)

    def resample_to_ref(self, ref, interp='auto', outside_val=None, in_place=False):
        if interp == 'auto':
            if self.min() >= 0 and self.max() < 255 and 'f' not in str(self.dtype):
                #  This should be a label
                interp = sitk.sitkNearestNeighbor
            else:
                # This should be an image
                interp = sitk.sitkBSpline
        if outside_val is None:
            outside_val = self.min()
        image = self.sitk_image
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(interp)
        resampleFilter.SetDefaultPixelValue(outside_val)
        ref_type = get_image_type(ref)
        if ref_type == 'smartimage' and not ref.loaded:
            ref = ref.path
            ref_type = 'path'
        if ref_type == 'smartimage':
            resampleFilter.SetReferenceImage(ref.sitk_image)
        elif ref_type == 'sitk':
            resampleFilter.SetReferenceImage(ref)
        elif ref_type in ['goudapath', 'path', 'string']:
            ref = str(ref)
            if os.path.isdir(ref):
                dicom_files = sorted(glob.glob(os.path.join(ref, '*.dcm')))
                image_path = dicom_files[0]
            else:
                image_path = ref
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(image_path)
            file_reader.ReadImageInformation()
            resampleFilter.SetSize(file_reader.GetSize())
            resampleFilter.SetOutputOrigin(file_reader.GetOrigin())
            resampleFilter.SetOutputSpacing(file_reader.GetSpacing())
            resampleFilter.SetOutputDirection(file_reader.GetDirection())
        elif ref_type == 'sitkreader':
            resampleFilter.SetSize(ref.GetSize())
            resampleFilter.SetOutputOrigin(ref.GetOrigin())
            resampleFilter.SetOutputSpacing(ref.GetSpacing())
            resampleFilter.SetOutputDirection(ref.GetDirection())
        elif ref_type == 'dict':
            resampleFilter.SetSize([int(item) for item in ref['size']])
            resampleFilter.SetOutputOrigin(ref['origin'])
            resampleFilter.SetOutputSpacing(ref['spacing'])
            resampleFilter.SetOutputDirection(ref['direction'])
        elif ref_type == 'smartimage':
            resampleFilter.SetSize(ref.GetSize().tolist())
            resampleFilter.SetOutputOrigin(ref.GetOrigin())
            resampleFilter.SetOutputSpacing(ref.GetSpacing())
            resampleFilter.SetOutputDirection(ref.GetDirection())
        else:
            raise ValueError("Unknown reference type: '{}'".format(type(ref)))
        result = resampleFilter.Execute(image)
        if in_place:
            self.update(result)
            return self
        else:
            return SmartImage(result)

    # Comparison Operators
    def __add__(self, other):
        return self.__perform_op(sitk.Add, itk.AddImageFilter, other, in_place=False, autocast=True)

    def __iadd__(self, other):
        self.__perform_op(sitk.Add, itk.AddImageFilter, other, in_place=True)
        # self.update(result)  # TODO - check that inplace ops are working
        return self

    def __eq__(self, other):
        image = self.image
        if self.image_type == 'sitk':
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__eq__(other))
        elif self.image_type == 'itk':
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __gt__(self, other):
        image = self.image
        if self.image_type == 'sitk':
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__gt__(other))
        elif self.image_type == 'itk':
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __lt__(self, other):
        image = self.image
        if self.image_type == 'sitk':
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__lt__(other))
        elif self.image_type == 'itk':
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __mul__(self, other):
        return self.__perform_op(sitk.Multiply, itk.MultiplyImageFilter, other, in_place=False)

    # def __rmul__(self, other):
        # This would be [other * self] rather than [self * other] - do we want this?

    def __truediv__(self, other):
        return self.__perform_op(sitk.Divide, itk.DivideImageFilter, other, in_place=False)

    def __ne__(self, other):
        image = self.image
        if self.image_type == 'sitk':
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__ne__(other))
        elif self.image_type == 'itk':
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __sub__(self, other):
        return self.__perform_op(sitk.Subtract, itk.SubtractImageFilter, other, in_place=False, autocast=True)

    def __isub__(self, other):
        self.__perform_op(sitk.Subtract, itk.SubtractImageFilter, other, in_place=True)
        # self.update(result)
        return self

    def __and__(self, other):
        return self.__perform_op(sitk.And, itk.AndImageFilter, other, in_place=False, autocast='uint8')

    def binary_and(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return self.__perform_op(sitk.Or, itk.OrImageFilter, other, in_place=False, autocast='uint8')

    def binary_or(self, other):
        return self.__or__(other)

    def change_label(self, changeMap: dict, in_place: bool = False):
        # TODO - add itk version (may need a internal method to run a filter...)
        return self.__perform_op(sitk.ChangeLabel, None, changeMap=changeMap, in_place=in_place)

    def __perform_op(self, sitk_op, itk_op, *args, in_place=False, autocast=False, **kwargs):
        """Perform the sitk/itk operation depending on the current image type

        Parameters
        ----------
        sitk_op : function
            The sitk operation - ex. sitk.Add, sitk.Subtract
        itk_op : function
            The itk operation - ex. itk.AddImageFilter, sitk.SubtractImageFilter
        in_place : bool
            If true, update the current image, by default False
        autocast : Union[bool, str]
            Can be true to detect output type based on :func:`numpy.result_type`, a string to manually set type, or False to not cast, by default False

        Note
        ----
        Any SmartImage or Image objects passed as args will be converted before being passed to the op.

        """
        image = self.image
        output_type = self.dtype
        if self.image_type == 'sitk':
            if sitk_op is None:
                raise NotImplementedError("The SimpleITK version of this operation has not been implemented for SmartImage yet")
            new_args = []
            for item in args:
                if get_image_type(item) in ['smartimage', 'itk']:
                    item = as_image(item).sitk_image
                    output_type = np.result_type(output_type, get_dtype(item))
                new_args.append(item)
            if autocast:
                output_type = SmartType.as_sitk(autocast if isinstance(autocast, str) else output_type)
                for idx in range(len(new_args)):
                    if get_image_type(new_args[idx]) == 'sitk':
                        new_args[idx] = sitk.Cast(new_args[idx], output_type)
                image = sitk.Cast(image, output_type)
            result = sitk_op(image, *new_args, **kwargs)
        elif self.image_type == 'itk':
            if itk_op is None:
                raise NotImplementedError("The ITK version of this operation has not been implemented for SmartImage yet")
            new_args = []
            for item in args:
                if get_image_type(item) in ['smartimage', 'sitk']:
                    item = as_image(item).itk_image
                    output_type = np.result_type(output_type, get_dtype(item))
                new_args.append(item)
            if autocast:
                output_type = SmartType.as_itk(autocast if isinstance(autocast, str) else output_type)
                for idx in range(len(new_args)):
                    if get_image_type(new_args[idx]) == 'itk':
                        new_args[idx] = itk.CastImageFilter[new_args[idx], output_type].New()(new_args[idx])
                image = itk.CastImageFilter[image, output_type].New()(image)
            result = itk_op(image, *new_args, **kwargs)

        result_type = get_image_type(result)
        if result_type in ['sitk', 'itk', 'smartimage']:
            if in_place:
                return self.update(result)
            else:
                return as_image(result)
        else:
            if in_place:
                warnings.warn('Could not update in-place as result is not an image type')
            return result

    def __setitem__(self, key, val):
        image = self.image
        if self.image_type == 'sitk' or self.image_type == 'itk':
            image[key] = val
        else:
            # Should never throw this error
            raise ValueError('Unknown image type: {}'.format(type(image)))

    def __getitem__(self, key):
        image = self.image
        if self.image_type == 'sitk' or self.image_type == 'itk':
            result = image[key]
            image_type = get_image_type(result)
            if image_type in ['itk', 'sitk', 'smartimage']:
                return as_image(result)
        else:
            # Should never throw this error
            raise ValueError('Unknown image type: {}'.format(type(image)))

    def apply(self, op, *args, in_place=True, **kwargs):
        """Apply an operation to the image

        Parameters
        ----------
        method : function
            The operation to apply to the image - should be the same type as the default image
        """
        result = op(self.image, *args, **kwargs)
        if in_place:
            self.update(result)
            return self
        return result


ImageType = Union[SmartImage, itk.Image, sitk.Image]
ImageArrayType = Union[SmartImage, itk.Image, sitk.Image, npt.NDArray]
ImageRefType = Union[SmartImage, itk.Image, sitk.Image, dict, sitk.ImageFileReader]


def as_image(image: ImageArrayType) -> SmartImage:
    """Wrap an image as a SmartImage"""
    if not isinstance(image, SmartImage):
        return SmartImage(image)
    else:
        return image


def as_image_type(image: ImageArrayType, output_type: str) -> ImageArrayType:
    """Convert an image to a specific type"""
    if output_type == 'sitk':
        return as_image(image).sitk_image
    elif output_type == 'itk':
        return as_image(image).itk_image
    elif output_type == 'numpy':
        return as_image(image).as_array()
    elif output_type == 'smartimage':
        return as_image(image)
    else:
        raise ValueError('Invalid image type: {}'.format(output_type))


def zeros_like(image: Union[itk.Image, sitk.Image, SmartImage, dict], dtype: Optional[str] = None):
    """Return an image of zeros with the same physical parameters as the input

    Parameters
    ----------
    image : Union[itk.Image, sitk.Image, SmartImage, dict]
        The image to copy physical parameters from
    dtype : str, optional
        The dtype to use instead of the reference image dtype, by default None
    """
    if not isinstance(image, dict):
        image = SmartImage(image).get_physical_properties()
    if dtype is not None:
        image['dtype'] = dtype
    if isinstance(image['size'], np.ndarray):
        image['size'] = tuple(image['size'].tolist())
    zero_image = sitk.Image(image['size'], SmartType.as_sitk(image['dtype']))
    zero_image.SetOrigin(image['origin'])
    zero_image.SetSpacing(image['spacing'])
    zero_image.SetDirection(image['direction'])
    return SmartImage(zero_image)

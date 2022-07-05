import glob
import os
import warnings
from typing import List, Optional, Union

import gouda
import itk
import numpy as np
import SimpleITK as sitk

from .constants import DTYPE_MATCH_ITK, DTYPE_MATCH_SITK, DTYPE_STRING

# DTYPE_MATCH_ITK = {
#     np.float32: itk.F,
#     np.float64: itk.D,
#     np.uint8: itk.UC,
#     np.int16: itk.SS,
#     np.uint16: itk.US,
# }
#
# DTYPE_MATCH_SITK = {
#     np.float32: sitk.sitkFloat32,
#     np.float64: sitk.sitkFloat64,
#     np.uint8: sitk.sitkUInt8,
#     np.int16: sitk.sitkInt16,
#     np.uint16: sitk.sitkUInt16,
# }


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


class SmartImage(object):
    def __init__(self, path, default_type='sitk'):
        self.__sitk_image: sitk.Image = None
        self.__itk_image: itk.Image = None
        self.__updated_itk: bool = False
        self.__updated_sitk: bool = False
        # These are calculated only as needed
        self.__reset_internals()

        self.default_type = default_type
        if hasattr(path, 'abspath'):
            path = path.abspath
        if not isinstance(path, (str, os.PathLike)):
            if isinstance(path, itk.Image):
                self.__itk_image = path
                self.__sitk_image = None
                self.__updated_itk = True
                self.path = ""
            elif isinstance(path, sitk.Image):
                self.__sitk_image = path
                self.__itk_image = None
                self.__updated_sitk = True
                self.path = ""
            elif isinstance(path, SmartImage):
                # NOTE: This is just a shallow copy of the other SmartImage for convenience
                self.default_type = path.default_type
                self.__sitk_image = path.__sitk_image
                self.__itk_image = path.__itk_image
                self.__updated_itk = path.__updated_itk
                self.__updated_sitk = path.__updated_sitk
                self.path = path.path
            elif isinstance(path, np.ndarray):
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
        if isinstance(image, sitk.Image):
            filt = sitk.StatisticsImageFilter()
            filt.Execute(image)
            self.__maximum_intensity = filt.GetMaximum()
            self.__minimum_intensity = filt.GetMinimum()
            self.__mean_intensity = filt.GetMean()
            self.__sum_intensity = filt.GetSum()
            self.__stddev_intensity = filt.GetSigma()
            self.__var_intensity = filt.GetVariance()
        elif isinstance(image, itk.Image):
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
        image = self.image
        if isinstance(image, itk.Image):
            # return type(image)  # returns the python type?
            return itk.template(image)[1][0]  # returns the C-type
        elif isinstance(image, sitk.Image):
            return image.GetPixelIDTypeAsString()

    @property
    def image(self):
        if self.__sitk_image is None and self.__itk_image is None:
            if self.default_type == 'sitk':
                return self.sitk_image
            elif self.default_type == 'itk':
                return self.itk_image
            else:
                raise ValueError("Unknown default image type: {}".format(self.default_type))
        if (self.__sitk_image is None or self.__updated_itk) != (self.__itk_image is None or self.__updated_sitk):
            return self.__sitk_image if self.__sitk_image is not None else self.__itk_image
        elif self.default_type == 'sitk':
            return self.sitk_image
        elif self.default_type == 'itk':
            return self.itk_image
        else:
            raise ValueError("No possible images to return...")

    @property
    def itk_image(self):
        if self.__updated_sitk:
            self.__reset_internals()
            self.__itk_image = self.__sitk2itk(self.sitk_image)
            self.__updated_sitk = False
        if self.__itk_image is None:
            self.__itk_image = itk.imread(str(self.path))
        return self.__itk_image

    @property
    def sitk_image(self):
        if self.__updated_itk:
            self.__reset_internals()
            self.__sitk_image = self.__itk2sitk(self.itk_image)
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
        return self.__sitk_image is None and self.__itk_image is None

    def __sitk2itk(self, image):
        """Local version of the method in convert to avoid circular imports"""
        if isinstance(image, itk.Image):
            return image
        itk_image = itk.GetImageFromArray(sitk.GetArrayViewFromImage(image), is_vector=image.GetNumberOfComponentsPerPixel() > 1)
        itk_image.SetOrigin(image.GetOrigin())
        itk_image.SetSpacing(image.GetSpacing())
        itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [image.GetDimension()] * 2)))
        return itk_image

    def __itk2sitk(self, image):
        """Local version of the method in convert to avoid circular imports"""
        if isinstance(image, sitk.Image):
            return image
        sitk_image = sitk.GetImageFromArray(itk.GetArrayViewFromImage(image), isVector=image.GetNumberOfComponentsPerPixel() > 1)
        sitk_image.SetOrigin(tuple(image.GetOrigin()))
        sitk_image.SetSpacing(tuple(image.GetSpacing()))
        sitk_image.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
        return sitk_image

    @property
    def ndim(self):
        image = self.image
        if isinstance(image, sitk.Image):
            return image.GetDimension()
        elif isinstance(image, itk.Image):
            return image.GetImageDimension()

    def as_array(self):
        image = self.image
        if isinstance(image, itk.Image):
            return itk.GetArrayFromImage(image)
        elif isinstance(image, sitk.Image):
            return sitk.GetArrayFromImage(image)
        else:
            raise ValueError("This should never reach here")

    def as_view(self):
        image = self.image
        if isinstance(image, itk.Image):
            return itk.GetArrayViewFromImage(image)
        elif isinstance(image, sitk.Image):
            return sitk.GetArrayViewFromImage(image)
        else:
            raise ValueError("This should never reach here")

    def astype(self, dtype, in_place=False, image_type=None):
        image_type = self.default_type if image_type is None else image_type
        if image_type == 'sitk':
            image = self.sitk_image
            if dtype in DTYPE_MATCH_SITK:
                dtype = DTYPE_MATCH_SITK[dtype]
            elif dtype in DTYPE_STRING:
                dtype = DTYPE_STRING[dtype][1]
            else:
                raise ValueError("Unknown dtype: {}".format(dtype))
            image = sitk.Cast(image, dtype)
            if in_place:
                return self.update(image)
            # cast image
        elif image_type == 'itk':
            image = self.itk_image
            if dtype in DTYPE_MATCH_ITK:
                dtype = DTYPE_MATCH_ITK[dtype]
            elif dtype in DTYPE_STRING:
                dtype = DTYPE_STRING[dtype][0]
            else:
                raise ValueError("Unknown dtype: {}".format(dtype))
            dim = image.GetImageDimension()
            caster = itk.CastImageFilter[type(image), itk.Image[dtype, dim]]
            caster.SetInput(image)
            caster.Update()
            image = caster.GetOutput()
            if in_place:
                return self.update(image)
        else:
            raise ValueError('Unknown image type: {}'.format(image_type))
        return image

    def GetDirection(self):
        image = self.image
        if isinstance(image, sitk.Image):
            return np.array(image.GetDirection())
        elif isinstance(image, itk.Image):
            return itk.GetArrayFromMatrix(image.GetDirection()).flatten()

    def SetDirection(self, direction):
        image = self.image
        if isinstance(image, sitk.Image):
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
        elif isinstance(image, itk.Image):
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
        if isinstance(image, sitk.Image):
            direction = np.array(image.GetDirection())
            nrows = int(np.sqrt(direction.shape[0]))
            return direction.reshape([nrows, -1])
        elif isinstance(image, itk.Image):
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
        return np.array(self.image.GetOrigin())

    def GetSize(self):
        image = self.image
        if isinstance(image, sitk.Image):
            return np.array(image.GetSize())
        elif isinstance(image, itk.Image):
            return np.array(image.GetLargestPossibleRegion().GetSize())

    def GetSpacing(self):
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

        if isinstance(image, sitk.Image):
            self.__updated_sitk = True
        elif isinstance(image, itk.Image):
            self.__updated_itk = True
        else:
            raise ValueError('Should never be here')
        return self
    
    #TODO - allow setting each physical value

    def unique(self):
        if self.__unique_labels is None:
            label_filt = sitk.LabelShapeStatisticsImageFilter()
            label_filt.Execute(self.sitk_image)
            self.labels = label_filt.GetLabels()
        return self.labels

    def update(self, image):
        if isinstance(image, sitk.Image):
            self.__sitk_image = image
            self.__updated_sitk = True
        elif isinstance(image, itk.Image):
            self.__itk_image = image
            self.__updated_itk = True
        self.__reset_internals()
        return self

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

    def window(self, min=None, max=None, level=None, width=None, in_place=False):
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
        
        image = self.image
        if isinstance(image, sitk.Image):
            result = sitk.IntensityWindowing(image, windowMinimum=min, windowMaximum=max, outputMinimum=min, outputMaximum=max)
        elif isinstance(image, itk.Image):
            filter = itk.IntensityWindowingImageFilter.New(image)
            filter.SetInput(image)
            filter.SetOutputMaximum(max)
            filter.SetOutputMinimum(min)
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
                 presmooth: bool = None,
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
        presmooth : bool | None | float
            Whether to apply a gaussian smoothing before resampling - if None, will use a sigma of 1 if the output size is smaller than the input size in all dimensions - if float, will use that as the smoothing sigma
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
        if isinstance(ref, SmartImage) and not ref.loaded:
            ref = ref.path
        if isinstance(ref, SmartImage):
            resampleFilter.SetReferenceImage(ref.sitk_image)
        elif isinstance(ref, sitk.Image):
            resampleFilter.SetReferenceImage(ref)
        elif isinstance(ref, (str, gouda.GoudaPath)):
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
        elif isinstance(ref, (sitk.ImageFileReader)):
            resampleFilter.SetSize(ref.GetSize())
            resampleFilter.SetOutputOrigin(ref.GetOrigin())
            resampleFilter.SetOutputSpacing(ref.GetSpacing())
            resampleFilter.SetOutputDirection(ref.GetDirection())
        elif isinstance(ref, dict):
            resampleFilter.SetSize(ref['size'])
            resampleFilter.SetOutputOrigin(ref['origin'])
            resampleFilter.SetOutputSpacing(ref['spacing'])
            resampleFilter.SetOutputDirection(ref['direction'])
        elif isinstance(ref, SmartImage):
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
        return self.__perform_op(sitk.Add, itk.AddImageFilter, other, in_place=False)

    def __iadd__(self, other):
        result = self.__perform_op(sitk.Add, itk.AddImageFilter, other, in_place=True)
        self.update(result)

    def __eq__(self, other):
        image = self.image
        if isinstance(image, sitk.Image):
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__eq__(other))
        elif isinstance(image, itk.Image):
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __gt__(self, other):
        image = self.image
        if isinstance(image, sitk.Image):
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__gt__(other))
        elif isinstance(image, itk.Image):
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __lt__(self, other):
        image = self.image
        if isinstance(image, sitk.Image):
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__lt__(other.sitk_image))
        elif isinstance(image, itk.Image):
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __ne__(self, other):
        image = self.image
        if isinstance(image, sitk.Image):
            if isinstance(other, SmartImage):  # unwrap other as-needed
                other = other.sitk_image
            return SmartImage(image.__ne__(other.sitk_image))
        elif isinstance(image, itk.Image):
            raise ValueError("Comparison operators are not supported yet for itk")
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __sub__(self, other):
        return self.__perform_op(sitk.Subtract, itk.SubtractImageFilter, other, in_place=False)

    def __isub__(self, other):
        result = self.__perform_op(sitk.Subtract, itk.SubtractImageFilter, other, in_place=True)
        self.update(result)
        
    def __and__(self, other):
        return self.__perform_op(sitk.And, itk.AndImageFilter, other, in_place=False)
        
    def __or__(self, other):
        return self.__perform_op(sitk.Or, itk.OrImageFilter, other, in_place=False)

    def __perform_op(self, sitk_op, itk_op, target, in_place=False):
        """Perform one of two operations depending on current image type

        Parameters
        ----------
        sitk_op : function
            The sitk operation - ex. sitk.Add, sitk.Subtract
        itk_op : function
            The itk operation - ex. itk.AddImageFilter, sitk.SubtractImageFilter
        target : anything
            Whatever the second argument of the operation will be

        Note
        ----
        All operations are assumed to have the forms `sitk.Operation(self.image, target)` or `itk.Operation(self.image, target)`
        """
        image = self.image
        if isinstance(image, sitk.Image):
            if isinstance(target, SmartImage):
                target = target.sitk_image
            result = sitk_op(image, target)
            if isinstance(result, sitk.Image) and not in_place:
                result = SmartImage(result)
            return result
        elif isinstance(image, itk.Image):
            if isinstance(target, SmartImage):
                target = target.itk_image
            result = itk_op(image, target)
            if isinstance(result, itk.Image) and not in_place:
                result = SmartImage(result)
            return result
        else:
            raise ValueError('self.image is type {}'.format(type(image)))

    def __getitem__(self, key):
        image = self.image
        if isinstance(image, sitk.Image):
            return SmartImage(image[key])
        elif isinstance(image, itk.Image):
            return SmartImage(image[key])
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
        print(op)
        print(self.image)
        print(args)
        print(kwargs)
        result = op(self.image, *args, **kwargs)
        if in_place:
            self.update(result)
            return self
        return result

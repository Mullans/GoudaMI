import warnings


class DummyModule:
    # TODO: Come up with a better way to handle missing itk
    def __init__(self, module_name):
        self.__module_name = module_name
        self.is_dummy = True
        self.Image = None
        self.SubtractImageFilter = None
        self.AndImageFilter = None
        self.OrImageFilter = None
        self.AddImageFilter = None
        warnings.filterwarnings('always', '.*(may prevent some methods).*')
        warnings.warn(f'ImportWarning: {self.__module_name} cannot be imported. This may prevent some methods from working as intended', RuntimeWarning, stacklevel=2)

    def imread(self, *args, **kwargs):
        warnings.warn(f'RuntimeWarning: called method requires {self.__module_name} to be installed', RuntimeWarning, stacklevel=2)

    def __getattr__(self, attr):
        warnings.warn(f'RuntimeWarning: called method requires {self.__module_name} to be installed', RuntimeWarning, stacklevel=2)


try:
    import itk
except ImportError:
    itk = DummyModule('itk')
except ModuleNotFoundError:
    itk = DummyModule('itk')

try:
    import vtk
except ImportError:
    vtk = DummyModule('vtk')
except ModuleNotFoundError:
    vtk = DummyModule('vtk')

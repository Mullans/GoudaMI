from setuptools import setup, find_namespace_packages

setup(name='GoudaMI',
      version='0.0.1',
      description='My common utilities for medical imaging analysis',
      url='',
      author='Sean Mullan',
      install_requires=[
          "SimpleITK",
          "numpy",
          "skimage",
          "gouda"
      ],
      keywords=['image segmentation', 'medical image analysis',
                'medical image segmentation'])

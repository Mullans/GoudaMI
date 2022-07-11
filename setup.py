import setuptools

VERSION = "0.0.2"
with open("requirements.txt") as reqf:
    required = reqf.read().splitlines()
    
setuptools.setup(
    name="GoudaMI",
    version=VERSION,
    author='Sean Mullan',
    author_email="sean-mullan@uiowa.edu",
    description='Some common utilities for medical imaging analysis',
    url='https://github.com/Mullans/GoudaMI',
    packages=setuptools.find_packages(exclude="tests"),
    keywords=['Dicom', 'Medical Imaging', 'Utilities', 'Segmentation', 'Registration'],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.6",
    install_requires=required
)
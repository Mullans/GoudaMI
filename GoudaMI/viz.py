import os

import SimpleITK as sitk


def get_viewer_path(viewer='Slicer'):
    """Get the location of a given viewer executable from the desktop entry"""
    if viewer == 'Slicer':
        entry_name = 'Slicer.desktop'
    elif viewer == 'ITKSnap' or viewer == 'Snap':
        entry_name = 'ITKSnap.desktop'
    else:
        raise ValueError("{} hasn't been added yet. Try 'Slicer' or 'ITKSnap' instead.")

    launcher_path = os.path.join(os.path.expanduser('~'), '.local/share/applications/', entry_name)
    if not os.path.exists(launcher_path):
        raise ValueError('No launcher path found for slicer. To use this method, create the file: {}'.format(launcher_path))
    with open(launcher_path, 'r') as infile:
        for line in infile:
            if 'Exec' in line:
                slicer_path = line[6:-2]
                break
    return slicer_path


def open_in_viewer(image, viewer='Slicer'):
    """Open an SIT"""
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    if os.path.sep not in viewer:
        viewer = get_viewer_path(viewer)
    image_viewer = sitk.ImageViewer()
    image_viewer.SetApplication(viewer)
    image_viewer.Execute(image)

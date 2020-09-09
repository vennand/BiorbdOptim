"""
Example script for animating a model
"""

from pathlib import Path

import numpy as np

from pyomeca import Markers, Rototrans, Angles
from BiorbdViz.biorbd_vtk import VtkModel, VtkWindow, Mesh

# Path to data
DATA_FOLDER = '/home/andre/Optimisation/data/DoCi/Essai/'
MARKERS_ANALOGS_C3D = DATA_FOLDER + "Do_44_mvtPrep_3.c3d"

# Load data
d = Markers.from_c3d(MARKERS_ANALOGS_C3D, usecols=range(0, 95), prefix_delimiter=":")

# Create a windows with a nice gray background
vtkWindow = VtkWindow(background_color=(0.5, 0.5, 0.5))

# Add marker holders to the window
vtkModelReal = VtkModel(vtkWindow, markers_color=(1, 0, 0), markers_size=10.0, markers_opacity=1, rt_length=100)
vtkModelPred = VtkModel(vtkWindow, markers_color=(0, 0, 0), markers_size=10.0, markers_opacity=0.5, rt_length=100)
vtkModelMid = VtkModel(vtkWindow, markers_color=(0, 0, 1), markers_size=10.0, markers_opacity=0.5, rt_length=100)
vtkModelFromC3d = VtkModel(vtkWindow, markers_color=(0, 1, 0), markers_size=10.0, markers_opacity=0.5, rt_length=100)

# Create some mesh (could be from any mesh source)
meshes = []
meshes.append(Mesh(vertex=d, triangles=[[0, 1], [5, 0], [1, 6]]))

# Animate all this
i = 0
while vtkWindow.is_active:
    # Update markers
    if i < 100:
        vtkModelReal.update_markers(d[:, :, i])
    else:
        # Dynamically change amount of markers for each Model
        vtkModelFromC3d.update_markers(d[:, :, i])

    # Funky online update of markers characteristics
    # if i > 150:
    #     vtkModelReal.set_markers_color(((i % 255.0) / 255.0, (i % 255.0) / 255.0, (i % 255.0) / 255.0))
    #     vtkModelFromC3d.set_markers_size((i % 150) / 20)

    # Update the meshing
    vtkModelReal.update_mesh([m[:, :, [i]] for m in meshes])

    # Update window
    vtkWindow.update_frame()
    i = (i + 1) % d.shape[2]
    print(i)
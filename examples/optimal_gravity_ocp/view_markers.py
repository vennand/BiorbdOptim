import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from load_data_filename import load_data_filename

from biorbd_optim import (
    OptimalControlProgram,
    Problem,
    Bounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
)

def rotating_gravity(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


# subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
subject = 'SaMi'
trial = '821_seul_1'
number_shooting_points = 100

data_path = '/home/andre/Optimisation/data/' + subject + '/'
model_path = data_path + 'Model/'
c3d_path = data_path + 'Essai/'
kalman_path = data_path + 'Q/'

data_filename = load_data_filename(subject, trial)
model_name = data_filename['model']
c3d_name = data_filename['c3d']
q_name = data_filename['q']
qd_name = data_filename['qd']
qdd_name = data_filename['qdd']
frames = data_filename['frames']

# --- Adjust number of shooting points --- #
list_adjusted_number_shooting_points = []
for frame_num in range(1, (frames.stop - frames.start - 1) // frames.step + 1):
    list_adjusted_number_shooting_points.append((frames.stop - frames.start - 1) // frame_num + 1)
diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
adjusted_number_shooting_points = ((frames.stop - frames.start - 1) // step_size + 1) - 1

biorbd_model = biorbd.Model(model_path + model_name)
c3d = ezc3d.c3d(c3d_path + c3d_name)
# q_ref = loadmat(kalman_path + q_name)['Q2']
# qdot_ref = loadmat(kalman_path + qd_name)['V2']
# qddot_ref = loadmat(kalman_path + qdd_name)['A2']
load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
ocp, sol = OptimalControlProgram.load(load_name + '.bo')
states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
q_ref = states['q']
rotating_gravity(biorbd_model, params["gravity_angle"].squeeze())

frequency = 200
# q_ref, qdot_ref, qddot_ref = correct_Kalman(q_ref, qdot_ref, qddot_ref, frequency)

biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

markers_mocap = c3d['data']['points'][:3, :95, frames.start:frames.stop:step_size] / 1000
c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
model_labels = [label.to_string() for label in biorbd_model.markerNames()]
labels_index = [index_c3d for label in model_labels for index_c3d, c3d_label in enumerate(c3d_labels) if label in c3d_label]
markers_reordered = np.zeros((3, len(labels_index), markers_mocap.shape[2]))
for index, label_index in enumerate(labels_index):
    markers_reordered[:, index, :] = markers_mocap[:, label_index, :]


markers = np.ndarray((3, markers_mocap.shape[1], q_ref.shape[1]))
symbolic_states = MX.sym("x", biorbd_model.nbQ(), 1)
markers_func = Function("ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]).expand()
for i in range(markers_mocap.shape[2]):
    markers[:, :, i] = markers_func(q_ref[:, i])

fig = pyplot.figure()
ax = Axes3D(fig)


for frame in range(markers.shape[2]):
    ax.scatter(markers[0, :, frame], markers[1, :, frame], markers[2, :, frame])
    ax.scatter(markers_reordered[0, :, frame], markers_reordered[1, :, frame], markers_reordered[2, :, frame])
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 2)
    pyplot.pause(0.01)
    pyplot.draw()
    ax.clear()
    print(frame)


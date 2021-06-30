import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import os
import pickle
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers

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
    gravity.applyRT(biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


# subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
subject = 'SaMi'
trial = '821_contact_1'
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
adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

biorbd_model = biorbd.Model(model_path + model_name)
c3d = ezc3d.c3d(c3d_path + c3d_name)

biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

### Matlab EKF ###
# q_ref = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size]
# qdot_ref = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
# qddot_ref = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]

### Biorbd EKF ###
# load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
# load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
# with open(load_variables_name, 'rb') as handle:
#     kalman_states = pickle.load(handle)
# q_ref = kalman_states['q'][:, ::step_size]

### OGE ###
# load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
# load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
# ocp, sol = OptimalControlProgram.load(load_name + '.bo')
# states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
# q_ref = states['q']
# gravity_angle = params["gravity_angle"].squeeze()
# rotating_gravity(biorbd_model, gravity_angle)

### OGE no OGE ###
load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_noOGE'
ocp, sol = OptimalControlProgram.load(load_name + '.bo')
states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
q_ref = states['q']

### OE ###
# load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
# load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
# load_variables_name = load_name + ".pkl"
# with open(load_variables_name, 'rb') as handle:
#     data = pickle.load(handle)
# q_ref = data['states']['q']
# gravity_angle = data['gravity_angle']
# rotating_gravity(biorbd_model, gravity_angle)

frequency = c3d['header']['points']['frame_rate']
# q_ref, qdot_ref, qddot_ref = correct_Kalman(q_ref, qdot_ref, qddot_ref, frequency)

markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)


markers = np.ndarray((3, markers_reordered.shape[1], q_ref.shape[1]))
symbolic_states = MX.sym("x", biorbd_model.nbQ(), 1)
markers_func = Function("ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]).expand()
for i in range(markers_reordered.shape[2]):
    markers[:, :, i] = markers_func(q_ref[:, i])

fig = pyplot.figure()
ax = Axes3D(fig)


for frame in range(markers.shape[2]):
    ax.scatter(markers[0, :, frame], markers[1, :, frame], markers[2, :, frame], color='blue', marker='x')
    ax.scatter(markers_reordered[0, :, frame], markers_reordered[1, :, frame], markers_reordered[2, :, frame], color='orange')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 2)
    pyplot.pause(0.1)
    pyplot.draw()
    ax.clear()
    print(frame)


import numpy as np
import scipy.optimize
import biorbd
# import BiorbdViz
import ezc3d
import pickle
import os
import sys
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
from load_data_filename import load_data_filename
from reorder_markers import reorder_markers


#
# This examples shows how to
#     1. Load a model
#     2. Generate data (should be acquired via real data)
#     3. Create a Kalman filter
#     4. Apply the Kalman filter (inverse kinematics)
#     5. Plot the kinematics (Q), velocity (Qdot) and acceleration (Qddot)
#
# Please note that this example will work only with the Eigen backend.
# Please also note that kalman will be VERY slow if compiled in debug
#

def rotating_gravity(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d().to_array()))
    biorbd_model.setGravity(gravity.to_array())

subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
# subject = 'SaMi'
trial = '822'
improve_initial_condition = True
extra_frames = 10
testing_angle = np.array([0, 10])

print('Subject: ', subject, ', Trial: ', trial)
print('Induced angle: ', testing_angle)

data_path = '/home/andre/Optimisation/data/' + subject + '/'
model_path = data_path + 'Model/'
c3d_path = data_path + 'Essai/'
kalman_path = data_path + 'Q/'

data_filename = load_data_filename(subject, trial)
model_name = data_filename['model']
c3d_name = data_filename['c3d']
frames = data_filename['frames']

biorbd_model = biorbd.Model(model_path + model_name)
c3d = ezc3d.c3d(c3d_path + c3d_name)

biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639).to_array())

optimal_gravity_filename_full = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(frames.stop - frames.start - 1) + '_mixed_EKF' + ".pkl"
with open(optimal_gravity_filename_full, 'rb') as handle:
    data = pickle.load(handle)
angle = data['params']["gravity_angle"].squeeze()
rotating_gravity(biorbd_model, angle)
rotating_gravity(biorbd_model, testing_angle*np.pi/180)

frames = range(frames.start-extra_frames, frames.stop, frames.step)

markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames)

markers_reordered[np.isnan(markers_reordered)] = 0.0  # Remove NaN

# Dispatch markers in biorbd structure so EKF can use it
markersOverFrames = []
for i in range(markers_reordered.shape[2]):
    markersOverFrames.append([biorbd.NodeSegment(m) for m in markers_reordered[:, :, i].T])

# Create a Kalman filter structure
frequency = c3d['header']['points']['frame_rate']  # Hz
# params = biorbd.KalmanParam(frequency=frequency, noiseFactor=1e-10, errorFactor=1e-5)
params = biorbd.KalmanParam(frequency=frequency)
kalman = biorbd.KalmanReconsMarkers(biorbd_model, params)

# Find an initial state to initialize Kalman
def states_to_markers(q):
    return np.array([marker.to_array() for marker in biorbd_model.markers(q)]).T

def distance_markers(q, *args):
    distances_ignoring_missing_markers = []
    markers_estimated = states_to_markers(q)
    for i in range(markers_reordered.shape[1]):
        if markers_reordered[0, i, 0] != 0:
            distances_ignoring_missing_markers.append(np.sqrt(np.sum((markers_estimated[:, i] - markers_reordered[:, i, 0])**2)))
    return np.sum(distances_ignoring_missing_markers)

if improve_initial_condition:
    Q_init = np.zeros(biorbd_model.nbQ())
    res = scipy.optimize.minimize(distance_markers, Q_init)
    Q_init = res.x

    kalman.setInitState(Q_init, np.zeros(biorbd_model.nbQ()), np.zeros(biorbd_model.nbQ()))

# Perform the kalman filter for each frame (the first frame is much longer than the next)
Q = biorbd.GeneralizedCoordinates(biorbd_model)
Qdot = biorbd.GeneralizedVelocity(biorbd_model)
Qddot = biorbd.GeneralizedAcceleration(biorbd_model)

q_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
qd_recons = np.ndarray((biorbd_model.nbQdot(), len(markersOverFrames)))
qdd_recons = np.ndarray((biorbd_model.nbQddot(), len(markersOverFrames)))
for i, targetMarkers in enumerate(markersOverFrames):
    kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
    q_recons[:, i] = Q.to_array()
    qd_recons[:, i] = Qdot.to_array()
    qdd_recons[:, i] = Qddot.to_array()


save_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_variable_gravity_ocp/Solutions/'
save_variables_name = save_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + "test_angle_" + str(testing_angle[1]) + ".pkl"
with open(save_variables_name, 'wb') as handle:
    pickle.dump({'q': q_recons[:, extra_frames:], 'qd': qd_recons[:, extra_frames:], 'qdd': qdd_recons[:, extra_frames:]},
                handle, protocol=3)

# Animate the results if biorbd viz is installed
# b = BiorbdViz.BiorbdViz(loaded_model=biorbd_model)
# b.load_movement(q_recons)
# b.exec()
import numpy as np
# import scipy
import biorbd
# import BiorbdViz
import ezc3d
import os
import pickle
from load_data_filename import load_data_filename


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

# subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
subject = 'SaMi'
trial = '821_822_4'

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

markers = c3d['data']['points'][:3, :95, frames.start:frames.stop]/1000  # XYZ1 x markers x time_frame
c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
model_labels = [label.to_string() for label in biorbd_model.markerNames()]
# labels_index = [index_c3d for label in model_labels for index_c3d, c3d_label in enumerate(c3d_labels) if label in c3d_label]

### --- Test --- ###
labels_index = []
missing_markers_index = []
for index_model, model_label in enumerate(model_labels):
    missing_markers_bool = True
    for index_c3d, c3d_label in enumerate(c3d_labels):
        if model_label in c3d_label:
            labels_index.append(index_c3d)
            missing_markers_bool = False
    if missing_markers_bool:
        labels_index.append(index_model)
        missing_markers_index.append(index_model)
### --- Test --- ###

markers_reordered = np.zeros((3, markers.shape[1], markers.shape[2]))
for index, label_index in enumerate(labels_index):
    if index in missing_markers_index:
        markers_reordered[:, index, :] = np.nan
    else:
        markers_reordered[:, index, :] = markers[:, label_index, :]
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


save_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
save_variables_name = save_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
with open(save_variables_name, 'wb') as handle:
    pickle.dump({'q': q_recons, 'qd': qd_recons, 'qdd': qdd_recons},
                handle, protocol=3)

# Animate the results if biorbd viz is installed
# b = BiorbdViz.BiorbdViz(loaded_model=biorbd_model)
# b.load_movement(q_recons)
# b.exec()
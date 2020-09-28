import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import pickle
import time
from casadi import MX, Function
import os
import warnings
from load_data_filename import load_data_filename
from x_bounds import x_bounds
from matplotlib import pyplot
from matplotlib.lines import Line2D




def check_Kalman(q_ref):
    segments_with_pi_limits = [15, 17, 18, 24, 26, 27, 31, 34, 37, 40]
    bool = np.zeros(q_ref.shape)
    for (i, j), q in np.ndenumerate(q_ref[6:, :]):
        if i+6+1 in segments_with_pi_limits:
            bool[i+6, j] = ((q / np.pi).astype(int) != 0)
        else:
            bool[i+6, j] = ((q / (np.pi/2)).astype(int) != 0)
    states_idx_bool = bool.any(axis=1)

    states_idx_range_list = []
    start_index = 0
    broken_dofs = []
    for idx, bool_idx in enumerate(states_idx_bool):
        if bool_idx:
            stop_index = idx
            if idx != start_index:
                states_idx_range_list.append(range(start_index, stop_index))
            start_index = stop_index + 1
            broken_dofs.append(stop_index)
    if bool.shape[0] != start_index:
        states_idx_range_list.append(range(start_index, bool.shape[0]))
    return states_idx_range_list, broken_dofs


def correct_Kalman(q_ref, qdot_ref, qddot_ref, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab):
    _, broken_dofs = check_Kalman(q_ref)
    _, broken_dofs_matlab = check_Kalman(q_ref_matlab)

    q_ref_corrected = q_ref_matlab
    qdot_ref_corrected = qdot_ref_matlab
    qddot_ref_corrected = qddot_ref_matlab

    for dof in broken_dofs_matlab:
        if dof not in broken_dofs:
            q_ref_corrected[dof, :] = q_ref[dof, :]
            qdot_ref_corrected[dof, :] = qdot_ref[dof, :]
            qddot_ref_corrected[dof, :] = qddot_ref[dof, :]

    return q_ref_corrected, qdot_ref_corrected, qddot_ref_corrected

subject = 'SaMi'
number_shooting_points = 100
trial = '821_seul_3'

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
q_ref_matlab = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop]
qdot_ref_matlab = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop]
qddot_ref_matlab = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop]


load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
with open(load_variables_name, 'rb') as handle:
    kalman_states = pickle.load(handle)
q_ref = kalman_states['q']
qdot_ref = kalman_states['qd']
qddot_ref = kalman_states['qdd']


# load_path_OGE = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
# load_variables_name = load_path_OGE + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(310) + '_matlab_EKF' + ".pkl"
# with open(load_variables_name, 'rb') as handle:
#     OGE_states = pickle.load(handle)
# q_ref_OGE = OGE_states['states']['q']
# # qdot_ref_OGE = OGE_states['states']['qdot']
# # qddot_ref_OGE = OGE_states['states']['qddot']
#
#
# load_path_OGE = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
# load_variables_name = load_path_OGE + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".pkl"
# with open(load_variables_name, 'rb') as handle:
#     OGE_mixed_states = pickle.load(handle)
# q_ref_OGE_mixed = OGE_mixed_states['states']['q']
# # qdot_ref_OGE_mixed = OGE_mixed_states['states']['qdot']
# # qddot_ref_OGE_mixed = OGE_mixed_states['states']['qddot']


dofs = [range(0, 6), range(6, 9), range(9, 12),
        range(12, 14), range(14, 17), range(17, 19), range(19, 21),
        range(21, 23), range(23, 26), range(26, 28), range(28, 30),
        range(30, 33), range(33, 34), range(34, 36),
        range(36, 39), range(39, 40), range(40, 42),
        ]
dofs_name = ['Pelvis', 'Thorax', 'Head',
             'Right shoulder', 'Right arm', 'Right forearm', 'Right hand',
             'Left shoulder', 'Left arm', 'Left forearm', 'Left hand',
             'Right thigh', 'Right leg', 'Right foot',
             'Left thigh', 'Left leg', 'Left foot',
             ]
# dofs = range(6, 9)
for idx_dof, dof in enumerate(dofs):
    fig = pyplot.figure()
    pyplot.plot(q_ref_matlab[dof, :].T, color='blue')
    pyplot.plot(q_ref[dof, :].T, color='red')
    # pyplot.plot(q_ref_OGE[dof, :].T, color='green')
    # pyplot.plot(range(0, 311, 3), q_ref_OGE_mixed[dof, :].T, color='orange')

    # fig = pyplot.figure()
    # pyplot.plot(qdot_ref_matlab[dof, :].T, color='blue')
    # pyplot.plot(qdot_ref[dof, :].T, color='red')
    #
    # fig = pyplot.figure()
    # pyplot.plot(qddot_ref_matlab[dof, :].T, color='blue')
    # pyplot.plot(qddot_ref[dof, :].T, color='red')

    pyplot.title(dofs_name[idx_dof])
    lm_matlab = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_biorbd = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    lm_OGE_mixed = Line2D([0, 1], [0, 1], linestyle='-', color='orange')
    pyplot.legend([lm_matlab, lm_biorbd, lm_OGE, lm_OGE_mixed], ['Matlab', 'Biorbd', 'OGE', 'OGE mixed'])


pyplot.show()
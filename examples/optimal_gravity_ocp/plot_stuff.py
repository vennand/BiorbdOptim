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
from adjust_number_shooting_points import adjust_number_shooting_points
from matplotlib import pyplot
from matplotlib.lines import Line2D




def check_Kalman(q_ref):
    segments_with_pi_limits = [14, 16, 17, 18, 23, 25, 26, 27, 30, 33, 36, 39]
    hand_segments = [19, 20, 28, 29]
    bool = np.zeros(q_ref.shape)
    bool[4, :] = ((q_ref[4, :] / (np.pi / 2)).astype(int) != 0)
    for (i, j), q in np.ndenumerate(q_ref[6:, :]):
        if i+6 in segments_with_pi_limits:
            bool[i+6, j] = ((q / np.pi).astype(int) != 0)
        elif i+6 in hand_segments:
            bool[i+6, j] = ((q / (3*np.pi/2)).astype(int) != 0)
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


def choose_Kalman(q_ref_1, qdot_ref_1, qddot_ref_1, q_ref_2, qdot_ref_2, qddot_ref_2):
    _, broken_dofs_1 = check_Kalman(q_ref_1)
    _, broken_dofs_2 = check_Kalman(q_ref_2)

    q_ref_chosen = np.copy(q_ref_1)
    qdot_ref_chosen = np.copy(qdot_ref_1)
    qddot_ref_chosen = np.copy(qddot_ref_1)

    for dof in broken_dofs_1:
        if dof not in broken_dofs_2:
            q_ref_chosen[dof, :] = q_ref_2[dof, :]
            qdot_ref_chosen[dof, :] = qdot_ref_2[dof, :]
            qddot_ref_chosen[dof, :] = qddot_ref_2[dof, :]

    return q_ref_chosen, qdot_ref_chosen, qddot_ref_chosen


def shift_by_pi(q, error_margin):
    if ((np.pi)*(1-error_margin)) < np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q - np.pi
    elif ((np.pi)*(1-error_margin)) < -np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q + np.pi

    return q


def correct_Kalman(biorbd_model, q):
    error_margin = 0.35
    q_corrected = np.copy(q)
    # q_corrected = q

    first_dof_segments_with_3DoFs = [6, 9, 14, 23, 30, 36]
    first_dof_segments_with_2DoFs = [12, 17, 19, 21, 26, 28, 34, 40]

    n_q = biorbd_model.nbQ()
    q_corrected[4, :] = q[4, :] - ((2 * np.pi) * (np.mean(q[4, :]) / (2 * np.pi)).astype(int))
    for dof in range(6, n_q):
        q_corrected[dof, :] = q[dof, :] - ((2*np.pi) * (np.mean(q[dof, :]) / (2*np.pi)).astype(int))
        if ((2*np.pi)*(1-error_margin)) < np.mean(q_corrected[dof, :]) < ((2*np.pi)*(1+error_margin)):
            q_corrected[dof, :] = q_corrected[dof, :] - (2*np.pi)
        elif ((2*np.pi)*(1-error_margin)) < -np.mean(q_corrected[dof, :]) < ((2*np.pi)*(1+error_margin)):
            q_corrected[dof, :] = q_corrected[dof, :] + (2*np.pi)

    if ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q_corrected[4, :])) < ((np.pi) * (1 + error_margin)):
        q_corrected[3, :] = shift_by_pi(q_corrected[3, :], error_margin)
        q_corrected[4, :] = -shift_by_pi(q_corrected[4, :], error_margin)
        q_corrected[5, :] = shift_by_pi(q_corrected[5, :], error_margin)

    for dof in first_dof_segments_with_2DoFs:
        if ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q_corrected[dof, :])) < ((np.pi) * (1 + error_margin)):
            q_corrected[dof, :] = shift_by_pi(q_corrected[dof, :], error_margin)
            q_corrected[dof+1, :] = -shift_by_pi(q_corrected[dof+1, :], error_margin)

    for dof in first_dof_segments_with_3DoFs:
        if (((np.pi) * (1 - error_margin)) < np.abs(np.mean(q_corrected[dof, :])) < ((np.pi) * (1 + error_margin)) or
            ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q_corrected[dof+1, :])) < ((np.pi) * (1 + error_margin)) or
            ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q_corrected[dof+2, :])) < ((np.pi) * (1 + error_margin))):
            q_corrected[dof, :] = shift_by_pi(q_corrected[dof, :], error_margin)
            q_corrected[dof+1, :] = -shift_by_pi(q_corrected[dof+1, :], error_margin)
            q_corrected[dof+2, :] = shift_by_pi(q_corrected[dof+2, :], error_margin)

    return q_corrected


# subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
subject = 'SaMi'
number_shooting_points = 100
trial = '821_822_5'

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
q_ref_matlab = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop]
qdot_ref_matlab = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop]
qddot_ref_matlab = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop]


load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
with open(load_variables_name, 'rb') as handle:
    kalman_states = pickle.load(handle)
q_ref_biorbd = kalman_states['q']
qdot_ref_biorbd = kalman_states['qd']
qddot_ref_biorbd = kalman_states['qdd']

# q_ref, qdot_ref, qddot_ref = correct_Kalman(q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab)


# load_path_OGE = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
# load_variables_name = load_path_OGE + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(310) + '_matlab_EKF' + ".pkl"
# with open(load_variables_name, 'rb') as handle:
#     OGE_states = pickle.load(handle)
# q_ref_OGE = OGE_states['states']['q']
# # qdot_ref_OGE = OGE_states['states']['qdot']
# # qddot_ref_OGE = OGE_states['states']['qddot']
#
#
load_path_OGE = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
load_variables_name = load_path_OGE + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".pkl"
if os.path.isfile(load_variables_name):
    with open(load_variables_name, 'rb') as handle:
        OGE_mixed_states = pickle.load(handle)
    q_ref_OGE_mixed = OGE_mixed_states['states']['q']
    # qdot_ref_OGE_mixed = OGE_mixed_states['states']['qdot']
    # qddot_ref_OGE_mixed = OGE_mixed_states['states']['qddot']
else:
    q_ref_OGE_mixed = None

q_corrected_matlab = correct_Kalman(biorbd_model, q_ref_matlab)
q_corrected_biorbd = correct_Kalman(biorbd_model, q_ref_biorbd)

# q_corrected_biorbd[3, :] = q_corrected_biorbd[3, :] - 4*np.pi
# q_corrected_biorbd[4, :] = -q_corrected_biorbd[4, :]
# q_corrected_biorbd[5, :] = q_corrected_biorbd[5, :] - 1.5*np.pi

q_corrected, _, _ = choose_Kalman(q_corrected_matlab, qdot_ref_matlab, qddot_ref_matlab, q_corrected_biorbd, qdot_ref_biorbd, qddot_ref_biorbd)
# q_corrected, _, _ = choose_Kalman(q_corrected_biorbd, qdot_ref_biorbd, qddot_ref_biorbd, q_corrected_matlab, qdot_ref_matlab, qddot_ref_matlab)

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
# dofs = range(0, 6)
for idx_dof, dof in enumerate(dofs):
    fig = pyplot.figure()
    # pyplot.plot(q_ref_matlab[dof, :].T, color='blue')
    # pyplot.plot(q_ref_biorbd[dof, :].T, color='red')
    # pyplot.plot(q_corrected_matlab[dof, :].T, color='blue')
    pyplot.plot(q_corrected_biorbd[dof, :].T, color='red')
    pyplot.plot(q_corrected[dof, :].T, linestyle='--', color='black')
    # pyplot.plot(q_ref_OGE[dof, :].T, color='green')
    if q_ref_OGE_mixed is not None:
        pyplot.plot(range(0, len(frames), step_size), q_ref_OGE_mixed[dof, :].T, color='orange')

    # fig = pyplot.figure()
    # pyplot.plot(qdot_ref_matlab[dof, :].T, color='blue')
    # pyplot.plot(qdot_ref_biorbd[dof, :].T, color='red')
    #
    # fig = pyplot.figure()
    # pyplot.plot(qddot_ref_matlab[dof, :].T, color='blue')
    # pyplot.plot(qddot_ref_biorbd[dof, :].T, color='red')

    pyplot.title(dofs_name[idx_dof])
    lm_matlab = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_biorbd = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    lm_OGE_mixed = Line2D([0, 1], [0, 1], linestyle='-', color='orange')
    pyplot.legend([lm_matlab, lm_biorbd, lm_OGE, lm_OGE_mixed], ['Matlab', 'Biorbd', 'OGE', 'OGE mixed'])

pyplot.show()

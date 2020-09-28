import numpy as np
import pickle
import biorbd
import BiorbdViz
import ezc3d
import os
from load_data_filename import load_data_filename


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


# subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
subject = 'SaMi'
trial = '821_seul_4'

data_path = '/home/andre/Optimisation/data/' + subject + '/'
model_path = data_path + 'Model/'
c3d_path = data_path + 'Essai/'
kalman_path = data_path + 'Q/'

data_filename = load_data_filename(subject, trial)
model_name = data_filename['model']
c3d_name = data_filename['c3d']

load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
with open(load_variables_name, 'rb') as handle:
    kalman_states = pickle.load(handle)

q_recons = kalman_states['q']
# qd_recons = kalman_states['qd']
# qdd_recons = kalman_states['qdd']

states_idx_range_list, broken_dofs = check_Kalman(q_recons)
if broken_dofs is not None:
    print('Abnormal Kalman states at DoFs: ', broken_dofs)

# Animate the results if biorbd viz is installed
b = BiorbdViz.BiorbdViz(model_path + model_name)
b.load_movement(q_recons)
b.exec()
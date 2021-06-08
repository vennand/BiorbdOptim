import biorbd
import matlab.engine
import numpy as np
from casadi import MX
import ezc3d
import pickle
from scipy.io import loadmat
import os
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from adjust_Kalman import shift_by_2pi


if __name__ == "__main__":
    subjects_trials = [('DoCi', '822', 100), ('DoCi', '44_1', 100), ('DoCi', '44_2', 100), ('DoCi', '44_3', 100),
                       ('BeLa', '44_1', 100), ('BeLa', '44_2', 80), ('BeLa', '44_3', 100),
                       ('GuSe', '44_2', 80), ('GuSe', '44_3', 100), ('GuSe', '44_4', 100),
                       ('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100), ('SaMi', '821_contact_3', 100), ('SaMi', '822_contact_1', 100),
                       ('SaMi', '821_seul_1', 100), ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_3', 100), ('SaMi', '821_seul_4', 100), ('SaMi', '821_seul_5', 100),
                       ('SaMi', '821_822_2', 100), ('SaMi', '821_822_3', 100),
                       ('JeCh', '833_1', 100), ('JeCh', '833_2', 100), ('JeCh', '833_3', 100), ('JeCh', '833_4', 100), ('JeCh', '833_5', 100),
                      ]

    diff_Q_EKF_OGE_44_all = []
    diff_Q_EKF_OGE_821_all = []
    diff_Q_EKF_OGE_822_all = []
    diff_Q_EKF_OGE_833_all = []

    diff_Q_EKF_OE_44_all = []
    diff_Q_EKF_OE_821_all = []
    diff_Q_EKF_OE_822_all = []
    diff_Q_EKF_OE_833_all = []

    diff_markers_EKF_44_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_44_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_821_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_821_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_822_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_822_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_833_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_EKF_833_sd_all = {'trunk': [], 'arms': [], 'legs': []}

    diff_markers_OGE_44_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_44_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_821_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_821_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_822_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_822_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_833_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_833_sd_all = {'trunk': [], 'arms': [], 'legs': []}

    diff_markers_OE_44_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_44_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_821_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_821_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_822_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_822_sd_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_833_all = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_833_sd_all = {'trunk': [], 'arms': [], 'legs': []}

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        if '44' in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_44_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_44_all
            diff_markers_EKF_trial_all = diff_markers_EKF_44_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_44_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_44_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_44_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_44_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_44_sd_all
        elif '821' in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_821_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_821_all
            diff_markers_EKF_trial_all = diff_markers_EKF_821_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_821_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_821_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_821_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_821_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_821_sd_all
        elif '822' in trial and '821' not in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_822_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_822_all
            diff_markers_EKF_trial_all = diff_markers_EKF_822_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_822_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_822_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_822_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_822_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_822_sd_all
        elif '833' in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_833_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_833_all
            diff_markers_EKF_trial_all = diff_markers_EKF_833_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_833_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_833_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_833_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_833_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_833_sd_all
        else:
            print('Unforseen trial')

        diff_Q_EKF_OGE_trial_all.append(data['diff_Q_EKF_OGE'])
        diff_Q_EKF_OE_trial_all.append(data['diff_Q_EKF_OE'])

        diff_markers_EKF_trial_all['trunk'].append(data['average_distance_between_markers_EKF_biorbd_trunk'])
        diff_markers_EKF_trial_sd_all['trunk'].append(data['sd_distance_between_markers_EKF_biorbd_trunk'])
        diff_markers_EKF_trial_all['arms'].append(data['average_distance_between_markers_EKF_biorbd_arms'])
        diff_markers_EKF_trial_sd_all['arms'].append(data['sd_distance_between_markers_EKF_biorbd_arms'])
        diff_markers_EKF_trial_all['legs'].append(data['average_distance_between_markers_EKF_biorbd_legs'])
        diff_markers_EKF_trial_sd_all['legs'].append(data['sd_distance_between_markers_EKF_biorbd_legs'])
        diff_markers_OGE_trial_all['trunk'].append(data['average_distance_between_markers_OGE_trunk'])
        diff_markers_OGE_trial_sd_all['trunk'].append(data['sd_distance_between_markers_OGE_trunk'])
        diff_markers_OGE_trial_all['arms'].append(data['average_distance_between_markers_OGE_arms'])
        diff_markers_OGE_trial_sd_all['arms'].append(data['sd_distance_between_markers_OGE_arms'])
        diff_markers_OGE_trial_all['legs'].append(data['average_distance_between_markers_OGE_legs'])
        diff_markers_OGE_trial_sd_all['legs'].append(data['sd_distance_between_markers_OGE_legs'])
        diff_markers_OE_trial_all['trunk'].append(data['average_distance_between_markers_OE_trunk'])
        diff_markers_OE_trial_sd_all['trunk'].append(data['sd_distance_between_markers_OE_trunk'])
        diff_markers_OE_trial_all['arms'].append(data['average_distance_between_markers_OE_arms'])
        diff_markers_OE_trial_sd_all['arms'].append(data['sd_distance_between_markers_OE_arms'])
        diff_markers_OE_trial_all['legs'].append(data['average_distance_between_markers_OE_legs'])
        diff_markers_OE_trial_sd_all['legs'].append(data['sd_distance_between_markers_OE_legs'])

    segment = ['Trunk', 'Arms', 'Legs']
    movement = ['44', '821', '822', '833']

    diff_Q_EKF_OGE_44 = [0.014498727, 0.094367067, 0.06050132]
    diff_Q_EKF_OGE_821 = [1.325024999, 1.083911115, 1.371477808]
    diff_Q_EKF_OGE_822 = [0.018954185, 0.074822258,  0.01302148]
    diff_Q_EKF_OGE_833 = [0.053259091, 0.238249955, 0.022015211]

    diff_Q_EKF_OE_44 = [0.118705711, 0.223486722, 0.20921395]
    diff_Q_EKF_OE_821 = [1.369255421, 1.153854187, 1.507596222]
    diff_Q_EKF_OE_822 = [0.116051528, 0.167908915, 0.123389499]
    diff_Q_EKF_OE_833 = [0.164200632, 0.716220609, 0.145015156]

    diff_markers_EKF_44 = [13.91107536, 13.88046581, 12.82408242]
    diff_markers_EKF_44_sd = [3.62004158, 4.054551365, 5.612577606]
    diff_markers_EKF_821 = [19.19946328, 24.43503531, 11.73282899]
    diff_markers_EKF_821_sd = [7.60155909, 14.84592044, 3.867380626]
    diff_markers_EKF_822 = [16.57596611, 19.52904062, 14.31614175]
    diff_markers_EKF_822_sd = [5.838689844, 10.90187775, 2.588686146]
    diff_markers_EKF_833 = [16.68327304, 16.72243615, 11.30048492]
    diff_markers_EKF_833_sd = [8.500877344, 15.12972291, 4.793912115]

    diff_markers_OGE_44 = [32.95870262, 35.8644495, 38.32950686]
    diff_markers_OGE_44_sd = [3.211682665, 4.013256999, 8.160416085]
    diff_markers_OGE_821 = [49.56115092, 98.08412794, 46.9824915]
    diff_markers_OGE_821_sd = [5.865123299, 75.54257484, 10.72275052]
    diff_markers_OGE_822 = [37.74973344, 39.22718513, 44.48209099]
    diff_markers_OGE_822_sd = [5.625708728, 10.0093352, 7.418972365]
    diff_markers_OGE_833 = [46.36820192, 66.62329486, 55.11830816]
    diff_markers_OGE_833_sd = [22.54801791, 47.87765886, 27.0075009]

    diff_markers_OE_44 = [36.82199834, 22.37470936, 23.56751524]
    diff_markers_OE_44_sd = [5.350310939, 7.0822335, 5.244400796]
    diff_markers_OE_821 = [46.75325213, 36.23973392, 28.82545557]
    diff_markers_OE_821_sd = [9.836173048, 21.98032118, 5.009719782]
    diff_markers_OE_822 = [40.69368133, 26.0974794, 23.44398985]
    diff_markers_OE_822_sd = [8.452370279, 14.86644623, 5.764998431]
    diff_markers_OE_833 = [50.17336191, 35.67402606, 25.50949352]
    diff_markers_OE_833_sd = [13.97821206, 8.612371814, 5.908106563]

    # Create dataset
    diff_Q = np.array(diff_Q_EKF_OGE_44 + diff_Q_EKF_OGE_821 + diff_Q_EKF_OGE_822 + diff_Q_EKF_OGE_833 +
                      diff_Q_EKF_OE_44 + diff_Q_EKF_OE_821 + diff_Q_EKF_OE_822 + diff_Q_EKF_OE_833)
    recons_type_Q = np.array(['EKF vs. OGE'] * 3 * 4 + ['EKF vs. OE'] * 3 * 4)
    movements_Q = np.array(list(np.repeat(movement, 3)) * 2)
    segments_Q = np.array(segment * 4 * 2)

    recons_ranges = [range(0, 12), range(12, 24)]
    data_Q = []
    for idx, recons_range in enumerate(recons_ranges):
        data_Q.append(pd.DataFrame(data=diff_Q[recons_range], index=range(diff_Q[recons_range].size), columns=['diff_Q']))
        data_Q[idx]['recons_type'] = recons_type_Q[recons_range]
        data_Q[idx]['movements'] = movements_Q[recons_range]
        data_Q[idx]['segments'] = segments_Q[recons_range]

    nb_44 = len(diff_Q_EKF_OGE_44_all)
    nb_821 = len(diff_Q_EKF_OGE_821_all)
    nb_822 = len(diff_Q_EKF_OGE_822_all)
    nb_833 = len(diff_Q_EKF_OGE_833_all)
    diff_Q_all = np.array(diff_Q_EKF_OGE_44_all + diff_Q_EKF_OGE_821_all + diff_Q_EKF_OGE_822_all + diff_Q_EKF_OGE_833_all +
                        diff_Q_EKF_OE_44_all + diff_Q_EKF_OE_821_all + diff_Q_EKF_OE_822_all + diff_Q_EKF_OE_833_all)
    recons_type_Q_all = np.array(['EKF vs. OGE'] * 26 + ['EKF vs. OE'] * 26)
    movements_Q_all = np.tile([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833, 2)

    data_Q_all = pd.DataFrame(data=diff_Q_all, index=range(diff_Q_all.size), columns=['diff_Q_all'])
    data_Q_all['std_all'] = None
    data_Q_all['recons_type'] = recons_type_Q_all
    data_Q_all['movements'] = movements_Q_all

    diff_markers = np.array(diff_markers_EKF_44 + diff_markers_EKF_821 + diff_markers_EKF_822 + diff_markers_EKF_833 +
                            diff_markers_OGE_44 + diff_markers_OGE_821 + diff_markers_OGE_822 + diff_markers_OGE_833 +
                            diff_markers_OE_44 + diff_markers_OE_821 + diff_markers_OE_822 + diff_markers_OE_833)
    diff_markers_sd = np.array(diff_markers_EKF_44_sd + diff_markers_EKF_821_sd + diff_markers_EKF_822_sd + diff_markers_EKF_833_sd +
                            diff_markers_OGE_44_sd + diff_markers_OGE_821_sd + diff_markers_OGE_822_sd + diff_markers_OGE_833_sd +
                            diff_markers_OE_44_sd + diff_markers_OE_821_sd + diff_markers_OE_822_sd + diff_markers_OE_833_sd)
    diff_markers_all = np.array(list(diff_markers_EKF_44_all.values()) + list(diff_markers_EKF_821_all.values()) + list(diff_markers_EKF_822_all.values()) + list(diff_markers_EKF_833_all.values()) +
                        list(diff_markers_OGE_44_all.values()) + list(diff_markers_OGE_821_all.values()) + list(diff_markers_OGE_822_all.values()) + list(diff_markers_OGE_833_all.values()) +
                        list(diff_markers_OE_44_all.values()) + list(diff_markers_OE_821_all.values()) + list(diff_markers_OE_822_all.values()) + list(diff_markers_OE_833_all.values()), dtype=object)
    diff_markers_sd = np.array([np.std(i) for i in diff_markers_all])
    diff_markers_sd_all = np.array(list(diff_markers_EKF_44_sd_all.values()) + list(diff_markers_EKF_821_sd_all.values()) + list(diff_markers_EKF_822_sd_all.values()) + list(diff_markers_EKF_833_sd_all.values()) +
                        list(diff_markers_OGE_44_sd_all.values()) + list(diff_markers_OGE_821_sd_all.values()) + list(diff_markers_OGE_822_sd_all.values()) + list(diff_markers_OGE_833_sd_all.values()) +
                        list(diff_markers_OE_44_sd_all.values()) + list(diff_markers_OE_821_sd_all.values()) + list(diff_markers_OE_822_sd_all.values()) + list(diff_markers_OE_833_sd_all.values()), dtype=object)
    recons_type_marker = np.array(['EKF'] * 3 * 4 + ['OGE'] * 3 * 4 + ['OE'] * 3 * 4)
    movements_marker = np.array(list(np.repeat(movement, 3)) * 3)
    segments_marker = np.array(segment * 4 * 3)

    recons_ranges = [range(0, 12), range(12, 24), range(24, 36)]
    data_marker = []
    for idx, recons_range in enumerate(recons_ranges):
        data_marker.append(pd.DataFrame(data=diff_markers[recons_range], index=range(diff_markers[recons_range].size), columns=['diff_markers']))
        data_marker[idx]['std'] = diff_markers_sd[recons_range]
        data_marker[idx]['diff_markers_all'] = diff_markers_all[recons_range]
        data_marker[idx]['std_all'] = diff_markers_sd_all[recons_range]
        data_marker[idx]['recons_type'] = recons_type_marker[recons_range]
        data_marker[idx]['movements'] = movements_marker[recons_range]
        data_marker[idx]['segments'] = segments_marker[recons_range]

    nb_44 = len(list(diff_markers_EKF_44_all.values())[0])
    nb_821 = len(list(diff_markers_EKF_821_all.values())[0])
    nb_822 = len(list(diff_markers_EKF_822_all.values())[0])
    nb_833 = len(list(diff_markers_EKF_833_all.values())[0])
    diff_markers_all = np.array(np.array(list(diff_markers_EKF_44_all.values()) + list(diff_markers_EKF_821_all.values()) + list(diff_markers_EKF_822_all.values()) + list(diff_markers_EKF_833_all.values()) +
                        list(diff_markers_OGE_44_all.values()) + list(diff_markers_OGE_821_all.values()) + list(diff_markers_OGE_822_all.values()) + list(diff_markers_OGE_833_all.values()) +
                        list(diff_markers_OE_44_all.values()) + list(diff_markers_OE_821_all.values()) + list(diff_markers_OE_822_all.values()) + list(diff_markers_OE_833_all.values()), dtype=object).sum())
    diff_markers_sd_all = np.array(np.array(list(diff_markers_EKF_44_sd_all.values()) + list(diff_markers_EKF_821_sd_all.values()) + list(diff_markers_EKF_822_sd_all.values()) + list(diff_markers_EKF_833_sd_all.values()) +
                        list(diff_markers_OGE_44_sd_all.values()) + list(diff_markers_OGE_821_sd_all.values()) + list(diff_markers_OGE_822_sd_all.values()) + list(diff_markers_OGE_833_sd_all.values()) +
                        list(diff_markers_OE_44_sd_all.values()) + list(diff_markers_OE_821_sd_all.values()) + list(diff_markers_OE_822_sd_all.values()) + list(diff_markers_OE_833_sd_all.values()), dtype=object).sum())
    recons_type_marker_all = np.array(['EKF'] * 3 * 26 + ['OGE'] * 3 * 26 + ['OE'] * 3 * 26)
    movements_marker_all = np.tile(np.repeat([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833, 3), 3)
    segments_marker_all = np.array((list(np.repeat(segment, nb_44)) + list(np.repeat(segment, nb_821)) + list(np.repeat(segment, nb_822)) + list(np.repeat(segment, nb_833))) * 3)

    recons_ranges = [range(0, 26*3), range(26*3, 26*3*2), range(26*3*2, 26*3*3)]
    data_marker_all = []
    for idx, recons_range in enumerate(recons_ranges):
        data_marker_all.append(pd.DataFrame(data=diff_markers_all[recons_range], index=range(diff_markers_all[recons_range].size), columns=['diff_markers_all']))
        data_marker_all[idx]['std_all'] = diff_markers_sd_all[recons_range]
        data_marker_all[idx]['recons_type'] = recons_type_marker_all[recons_range]
        data_marker_all[idx]['movements'] = movements_marker_all[recons_range]
        data_marker_all[idx]['segments'] = segments_marker_all[recons_range]

    patches_order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    clrs_bright = sns.color_palette("bright", 3)
    clrs_dark = sns.color_palette("dark", 3)
    alpha = 0.6

    clrs_hatches_EKF_vs_OGE = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_Q[0].loc[data_Q[0]['movements'] == y]['diff_Q']))
                               else (clrs_bright[idx], 'x')
                               for y in movement
                               for idx, x in enumerate(data_Q[0].loc[data_Q[0]['movements'] == y]['diff_Q'])]
    clrs_hatches_EKF_vs_OE = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_Q[1].loc[data_Q[1]['movements'] == y]['diff_Q']))
                              else (clrs_bright[idx], 'x')
                              for y in movement
                              for idx, x in enumerate(data_Q[1].loc[data_Q[1]['movements'] == y]['diff_Q'])]

    clrs_hatches_EKF = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_marker[0].loc[data_marker[0]['movements'] == y]['diff_markers']))
                        else (clrs_bright[idx], 'x')
                        for y in movement
                        for idx, x in enumerate(data_marker[0].loc[data_marker[0]['movements'] == y]['diff_markers'])]
    clrs_hatches_OGE = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_marker[1].loc[data_marker[1]['movements'] == y]['diff_markers']))
                        else (clrs_bright[idx], 'x')
                        for y in movement
                        for idx, x in enumerate(data_marker[1].loc[data_marker[1]['movements'] == y]['diff_markers'])]
    clrs_hatches_OE = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_marker[2].loc[data_marker[2]['movements'] == y]['diff_markers']))
                       else (clrs_bright[idx], 'x')
                       for y in movement
                       for idx, x in enumerate(data_marker[2].loc[data_marker[2]['movements'] == y]['diff_markers'])]

    clrs_hatches_EKF_vs_OGE[:] = [clrs_hatches_EKF_vs_OGE[i] for i in patches_order]
    clrs_hatches_EKF_vs_OE[:] = [clrs_hatches_EKF_vs_OE[i] for i in patches_order]

    clrs_hatches_EKF[:] = [clrs_hatches_EKF[i] for i in patches_order]
    clrs_hatches_OGE[:] = [clrs_hatches_OGE[i] for i in patches_order]
    clrs_hatches_OE[:] = [clrs_hatches_OE[i] for i in patches_order]


    def grouped_barplot(df, cat, subcat, val, err, all_val, ax_df):
        h_bar = []
        h_scatter = []
        u = df[cat].unique()
        x = np.arange(len(u))
        subx = df[subcat].unique()
        offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
        width = np.diff(offsets).mean()*0.9
        for i, gr in enumerate(subx):
            dfg = df[df[subcat] == gr]
            h_bar.append(ax_df.bar(x + offsets[i], dfg[val].values, width=width, label="{} {}".format(subcat, gr), yerr=dfg[err].values, color=clrs_bright[i], zorder=0))
            for x_i in x:
                h_scatter.append(ax_df.scatter([x_i + offsets[i]]*len(dfg[all_val].values[x_i]), dfg[all_val].values[x_i], color='black', zorder=1))
        ax_df.set_xlabel(df['recons_type'].unique().squeeze(), fontsize=14)
        ax_df.set_xticks(x)
        ax_df.set_xticklabels(u)
        ax_df.tick_params(axis='both', labelsize=12)
        return h_bar, h_scatter

    fig_marker_custom, axs_marker_custom = pyplot.subplots(1, 3, figsize=(20, 10), squeeze=False)
    df = data_marker
    cat = "movements"
    subcat = "segments"
    val = "diff_markers"
    err = "std"
    all_val = "diff_markers_all"
    h_group_barplot = []
    for i in range(3):
        h_group_barplot.append(grouped_barplot(df[i], cat, subcat, val, err, all_val, axs_marker_custom[0, i]))
    axs_marker_custom[0, -1].legend(h_group_barplot[-1][0], ['Trunk', 'Arms', 'Legs'], fontsize=14)
    axs_marker_custom[0, 0].set_ylabel('Marker error (mm)', fontsize=14)

    # Draw barplot
    # Joint angle Q difference
    fig_Q_all = pyplot.figure(figsize=(20, 10))
    axs_Q_all = fig_Q_all.gca()
    g_Q_all = sns.barplot(data=data_Q_all, x="movements", y="diff_Q_all", hue="recons_type", palette="bright", alpha=1, ax=axs_Q_all)
    g_Q_all.set_xlabel('Movements', fontsize=14)
    g_Q_all.set_ylabel('Joint angle diffenrence (°)', fontsize=14)
    g_Q_all.tick_params(labelsize=12)
    g_Q_all.legend(title="", fontsize=14)

    fig_Q, axs_Q = pyplot.subplots(1, 2, figsize=(20, 10), squeeze=False)
    g_EKF_vs_OGE = sns.barplot(data=data_Q[0], x="movements", y="diff_Q", hue="segments", palette="bright", alpha=1, ax=axs_Q[0, 0])
    g_EKF_vs_OGE.set_xlabel("EKF vs. OGE", fontsize=14)
    g_EKF_vs_OGE.set_ylabel("Joint angle difference (°)", fontsize=14)
    g_EKF_vs_OGE.tick_params(labelsize=12)
    g_EKF_vs_OGE.legend_.remove()
    for i, thisbar in enumerate(g_EKF_vs_OGE.patches):
        thisbar.set_color(clrs_hatches_EKF_vs_OGE[i][0])
        thisbar.set_edgecolor("white")

    g_EKF_vs_OE = sns.barplot(data=data_Q[1], x="movements", y="diff_Q", hue="segments", palette="bright", alpha=1, ax=axs_Q[0, 1])
    g_EKF_vs_OE.set_xlabel("EKF vs. OE", fontsize=14)
    g_EKF_vs_OE.set_ylabel("")
    g_EKF_vs_OE.tick_params(labelsize=12)
    g_EKF_vs_OE.legend(title="")
    for i, thisbar in enumerate(g_EKF_vs_OE.patches):
        thisbar.set_color(clrs_hatches_EKF_vs_OE[i][0])
        thisbar.set_edgecolor("white")

    # Marker error difference
    fig_marker, axs_marker = pyplot.subplots(1, 3, figsize=(20, 10), squeeze=False)
    # g_EKF_new = sns.barplot(data=data_marker_all[0], x="movements", y="diff_markers_all", hue="segments", ci="sd", palette="bright", alpha=1, ax=axs_marker[0, 0])
    g_EKF = sns.barplot(data=data_marker[0], x="movements", y="diff_markers", hue="segments", ci="sd", palette="bright", alpha=1, ax=axs_marker[0, 0])
    # g_EKF_swarm = sns.swarmplot(data=data_marker[0], x="movements", y="diff_markers", hue="segments", color="black", alpha=1, ax=axs_marker[0, 0])
    g_EKF.set_xlabel("EKF", fontsize=14)
    g_EKF.set_ylabel("Marker error (mm)", fontsize=14)
    g_EKF.tick_params(labelsize=12)
    g_EKF.legend_.remove()
    for i, thisbar in enumerate(g_EKF.patches):
        thisbar.set_color(clrs_hatches_EKF[i][0])
        thisbar.set_edgecolor("white")
        # thisbar.set_hatch(clrs_hatches_EKF[i][1])

    g_OGE = sns.barplot(data=data_marker[1], x="movements", y="diff_markers", hue="segments", ci="std", palette="bright", alpha=1, ax=axs_marker[0, 1])
    g_OGE.set_xlabel("OGE", fontsize=14)
    g_OGE.set_ylabel("")
    g_OGE.tick_params(labelsize=12)
    g_OGE.legend_.remove()
    for i, thisbar in enumerate(g_OGE.patches):
        thisbar.set_color(clrs_hatches_OGE[i][0])
        thisbar.set_edgecolor("white")
        # thisbar.set_hatch(clrs_hatches_OGE[i][1])

    g_OE = sns.barplot(data=data_marker[2], x="movements", y="diff_markers", hue="segments", ci="std", palette="bright", alpha=1, ax=axs_marker[0, 2])
    g_OE.set_xlabel("OE", fontsize=14)
    g_OE.set_ylabel("")
    g_OE.tick_params(labelsize=12)
    g_OE.legend(title="")
    for i, thisbar in enumerate(g_OE.patches):
        thisbar.set_color(clrs_hatches_OE[i][0])
        thisbar.set_edgecolor("white")
        # thisbar.set_hatch(clrs_hatches_OE[i][1])

    plot_ylimits = []
    plot_ylimits.append(g_EKF_vs_OGE.get_ylim()[1])
    plot_ylimits.append(g_EKF_vs_OE.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    g_EKF_vs_OGE.set(ylim=(0, plot_ymax))
    g_EKF_vs_OE.set(ylim=(0, plot_ymax))

    plot_ylimits = []
    plot_ylimits.append(g_EKF.get_ylim()[1])
    plot_ylimits.append(g_OGE.get_ylim()[1])
    plot_ylimits.append(g_OE.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    g_EKF.set(ylim=(0, plot_ymax))
    g_OGE.set(ylim=(0, plot_ymax))
    g_OE.set(ylim=(0, plot_ymax))

    plot_ylimits = []
    for axs in axs_marker_custom.squeeze():
        plot_ylimits.append(axs.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    pyplot.setp(axs_marker_custom, ylim=(0, plot_ymax))

    save_path = 'Solutions/'
    fig_Q_all.tight_layout
    save_name = save_path + 'Joint_angle_diff_barplot_all' + '.png'
    # fig_Q_all.savefig(save_name)

    fig_Q.tight_layout
    save_name = save_path + 'Joint_angle_diff_barplot' + '.png'
    # fig_Q.savefig(save_name)

    fig_marker.tight_layout
    save_name = save_path + 'Marker_error_barplot' + '.png'
    # fig_marker.savefig(save_name)

    fig_marker_custom.tight_layout
    save_name = save_path + 'Marker_error_barplot_custom' + '.png'
    # fig_marker_custom.savefig(save_name)

    pyplot.show()

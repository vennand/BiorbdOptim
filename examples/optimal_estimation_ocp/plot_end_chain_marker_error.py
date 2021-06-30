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
    subjects_trials = [('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100),
                       ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_4', 100),
                      ]

    segment_marker_error_OE_821 = []
    segment_marker_error_OGE_821 = []
    segment_marker_error_EKF_821 = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        segment_marker_error_OE_821.append(data['segment_marker_error_OE'])
        segment_marker_error_OGE_821.append(data['segment_marker_error_OGE'])
        segment_marker_error_EKF_821.append(data['segment_marker_error_EKF_biorbd'])


    subjects_trials = [('JeCh', '833_1', 100), ('JeCh', '833_2', 100), ('JeCh', '833_3', 100), ('JeCh', '833_4', 100),
                       ('JeCh', '833_5', 100),
                       ]

    segment_marker_error_OE_833 = []
    segment_marker_error_OGE_833 = []
    segment_marker_error_EKF_833 = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        segment_marker_error_OE_833.append(data['segment_marker_error_OE'])
        segment_marker_error_OGE_833.append(data['segment_marker_error_OGE'])
        segment_marker_error_EKF_833.append(data['segment_marker_error_EKF_biorbd'])

    # --- Plots --- #
    segments = ['Pelvis', 'Thorax', 'EpauleD', 'BrasD', 'ABrasD', 'MainD']

    fig = pyplot.figure(figsize=(20, 10))

    clrs_bright = sns.color_palette("bright", 2)

    for trial_OE, trial_OGE in zip(segment_marker_error_OE_821, segment_marker_error_OGE_821):
        for idx, segment in enumerate(segments):
            pyplot.plot(idx, trial_OE[segment], '-s', markersize=11, color=clrs_bright[0])
            pyplot.plot(idx, trial_OGE[segment], '-s', markersize=11, color=clrs_bright[1])
    # pyplot.xlabel('Segment', fontsize=16)
    pyplot.ylabel('Marker error (mm)', fontsize=16)
    pyplot.xticks(np.arange(len(segments)), segments, fontsize=14)
    pyplot.yticks(fontsize=14)
    fig.gca().legend(['Marker tracking 821<', 'Joint angle tracking 821<'], fontsize=15)

    print('EKF hands mean RMS 821: ', np.nanmean([trial['MainD'] for trial in segment_marker_error_EKF_821]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_EKF_821]))
    print('Joint angle tracking hands mean RMS 821: ', np.nanmean([trial['MainD'] for trial in segment_marker_error_OGE_821]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_OGE_821]))
    print('Marker tracking hands mean RMS 821: ', np.nanmean([trial['MainD'] for trial in segment_marker_error_OE_821]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_OE_821]))

    print('EKF hands mean RMS 833: ', np.nanmean([trial['MainG'] for trial in segment_marker_error_EKF_833]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_EKF_833]))
    print('Joint angle tracking hands mean RMS 833: ', np.nanmean([trial['MainG'] for trial in segment_marker_error_OGE_833]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_OGE_833]))
    print('Marker tracking hands mean RMS 833: ', np.nanmean([trial['MainG'] for trial in segment_marker_error_OE_833]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_OE_833]))

    save_path = 'Solutions/'
    save_name = save_path + "End_chain_marker_error" + '.png'
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    pyplot.show()


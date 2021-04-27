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
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from adjust_Kalman import shift_by_2pi


if __name__ == "__main__":
    subjects_trials = [((('DoCi', '44_1')), (('DoCi', '44_2'), ('DoCi', '44_3'))),
                       ((('BeLa', '44_3')), (('BeLa', '44_1'), ('BeLa', '44_3'))),
                       ((('SaMi', '821_contact_2')), (('SaMi', '821_contact_1'), ('SaMi', '821_contact_3'))),
                       ((('SaMi', '821_seul_3'), ('SaMi', '821_seul_4')), (('SaMi', '821_seul_1'), ('SaMi', '821_seul_2'), ('SaMi', '821_seul_5'))),
                       ((('SaMi', '821_822_2')), (('SaMi', '821_822_2'))),
                       ]

    subjects_trials = [('DoCi', '822', 100), ('DoCi', '44_1', 100), ('DoCi', '44_2', 100), ('DoCi', '44_3', 100),
                       ('BeLa', '44_1', 100), ('BeLa', '44_2', 80), ('BeLa', '44_3', 100),
                       ('GuSe', '44_2', 80), ('GuSe', '44_3', 100), ('GuSe', '44_4', 100),
                       ('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100), ('SaMi', '821_contact_3', 100), ('SaMi', '822_contact_1', 100),
                       ('SaMi', '821_seul_1', 100), ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_3', 100), ('SaMi', '821_seul_4', 100), ('SaMi', '821_seul_5', 100),
                       ('SaMi', '821_822_2', 100), ('SaMi', '821_822_3', 100),
                       ('JeCh', '833_1', 100), ('JeCh', '833_2', 100), ('JeCh', '833_3', 100), ('JeCh', '833_4', 100), ('JeCh', '833_5', 100),
                      ]

    load_path = 'Solutions/'
    load_name = load_path + 'meanfreq'
    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    MNF_EKF_matlab_sum = data['MNF']['EKF_matlab']
    MNF_EKF_biorbd_sum = data['MNF']['EKF_biorbd']
    MNF_OGE_sum = data['MNF']['OGE']
    MNF_OE_sum = data['MNF']['OE']

    # fig = pyplot.figure()
    # pyplot.plot(fft_freq[0], FFT_EKF_biorbd_sum[0], 's', color='blue', marker='o')
    # pyplot.plot(fft_freq[1], FFT_EKF_biorbd_sum[1], 's', color='blue', marker='X')
    # pyplot.axvline(MNF_EKF_biorbd_sum[0], color='blue', linestyle='-')
    # pyplot.axvline(MNF_EKF_biorbd_sum[1], color='blue', linestyle='--')
    #
    # pyplot.plot(fft_freq[0], FFT_OGE_sum[0], 's', color='red', marker='o')
    # pyplot.plot(fft_freq[1], FFT_OGE_sum[1], 's', color='red', marker='X')
    # pyplot.axvline(MNF_OGE_sum[0], color='red', linestyle='-')
    # pyplot.axvline(MNF_OGE_sum[1], color='red', linestyle='--')
    #
    # pyplot.plot(fft_freq[0], FFT_OE_sum[0], 's', color='green', marker='o')
    # pyplot.plot(fft_freq[1], FFT_OE_sum[1], 's', color='green', marker='X')
    # pyplot.axvline(MNF_OE_sum[0], color='green', linestyle='-')
    # pyplot.axvline(MNF_OE_sum[1], color='green', linestyle='--')
    #
    # pyplot.xlabel('Hz')
    # pyplot.title('Control frequency')
    #
    # lm_kalman_o = Line2D([0, 1], [0, 1], linestyle='', marker='o', color='blue')
    # lm_kalman_X = Line2D([0, 1], [0, 1], linestyle='', marker='X', color='blue')
    # lm_OGE_o = Line2D([0, 1], [0, 1], linestyle='', marker='o', color='red')
    # lm_OGE_X = Line2D([0, 1], [0, 1], linestyle='', marker='X', color='red')
    # lm_OE_o = Line2D([0, 1], [0, 1], linestyle='', marker='o', color='green')
    # lm_OE_X = Line2D([0, 1], [0, 1], linestyle='', marker='X', color='green')
    # pyplot.legend([lm_kalman_o, lm_kalman_X, lm_OGE_o, lm_OGE_X, lm_OE_o, lm_OE_X], ['EKF', 'EKF', 'OGE', 'OGE torq diff', 'OE', 'OE torq diff'])
    #
    # save_path = 'Solutions/'
    # save_name = save_path + subject + '/Plots/' + subject + '_' + trials[0] + '_' + trials[1] + "_FFT_controls" + '.png'
    # fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    # ax = sns.stripplot(x="population type", y="dollar_price", data=merged_df2, jitter=True)
    # ax.set_xlabel("Population type")
    # ax.set_ylabel("BigMac index (US$)")

    # Load the penguins dataset
    df = sns.load_dataset("penguins")
    # Draw a categorical scatterplot to show each observation
    ax = sns.swarmplot(data=df, x="body_mass_g", y="sex", hue="species")
    ax.set(ylabel="")

    # print('EKF MATLAB :', MNF_EKF_matlab_sum)
    # print('EKF biorbd :', MNF_EKF_biorbd_sum)
    # print('OGE :', MNF_OGE_sum)
    # print('OE :', MNF_OE_sum)

    pyplot.show()

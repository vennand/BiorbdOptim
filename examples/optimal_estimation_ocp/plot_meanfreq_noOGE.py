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
    # subjects_trials = [((('DoCi', '44_1')), (('DoCi', '44_2'), ('DoCi', '44_3'))),
    #                    ((('BeLa', '44_3')), (('BeLa', '44_1'), ('BeLa', '44_3'))),
    #                    ((('SaMi', '821_contact_2')), (('SaMi', '821_contact_1'), ('SaMi', '821_contact_3'))),
    #                    ((('SaMi', '821_seul_3'), ('SaMi', '821_seul_4')), (('SaMi', '821_seul_1'), ('SaMi', '821_seul_2'), ('SaMi', '821_seul_5'))),
    #                    ((('SaMi', '821_822_2')), (('SaMi', '821_822_2'))),
    #                    ]

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

    MNF_EKF_biorbd_mean = np.mean(MNF_EKF_biorbd_sum)
    MNF_EKF_biorbd_std = np.std(MNF_EKF_biorbd_sum)
    MNF_OGE_mean = np.mean(MNF_OGE_sum)
    MNF_OGE_std = np.std(MNF_OGE_sum)
    MNF_OE_mean = np.mean(MNF_OE_sum)
    MNF_OE_std = np.std(MNF_OE_sum)

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

    # Create dataset
    MNF = np.array(MNF_EKF_biorbd_sum + MNF_OGE_sum + MNF_OE_sum)
    recons_type = np.array(['EKF'] * len(MNF_EKF_biorbd_sum) + ['OGE'] * len(MNF_OGE_sum) + ['OE'] * len(MNF_OE_sum))
    subjects = np.array([subject_trial[0] for subject_trial in subjects_trials] * 3)
    trials = np.array([subject_trial[1] for subject_trial in subjects_trials] * 3)
    number_shooting_points = np.array([subject_trial[2] for subject_trial in subjects_trials] * 3)

    data = pd.DataFrame(data=MNF, index=range(MNF.size), columns=['MNF'])
    data['recons_type'] = recons_type
    data['subjects'] = subjects
    data['trials'] = trials
    data['number_shooting_points'] = number_shooting_points

    # Draw swarmplot
    fig = pyplot.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    ax = sns.swarmplot(data=data, x='recons_type', y='MNF', size=10)
    ax.set(xlabel="")
    # ax.set_ylim([0, 21])


    # # Create dataset for each movement
    # data_44 = data.loc[data['trials'].isin(['44_1', '44_2', '44_3', '44_4'])]
    # data_821 = data.loc[data['trials'].isin(['821_contact_1', '821_contact_2', '821_contact_3',
    #                                          '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5',
    #                                          '821_822_2', '821_822_3'])]
    # data_822 = data.loc[data['trials'].isin(['822', '822_contact_1'])]
    # data_833 = data.loc[data['trials'].isin(['833_1', '833_2', '833_3', '833_4', '833_5'])]
    #
    # # Draw swarmplot
    # fig_movement, axs_movement = pyplot.subplots(nrows=2, ncols=2, figsize=(20, 10))
    # sns.swarmplot(data=data_44, x='recons_type', y='MNF', ax=axs_movement[0, 0])
    # axs_movement[0, 0].set(xlabel="")
    # axs_movement[0, 0].set_title('44')
    # axs_movement[0, 0].set_ylim([0, 21])
    # sns.swarmplot(data=data_821, x='recons_type', y='MNF', ax=axs_movement[1, 0])
    # axs_movement[1, 0].set(xlabel="")
    # axs_movement[1, 0].set_title('821')
    # axs_movement[1, 0].set_ylim([0, 21])
    # sns.swarmplot(data=data_822, x='recons_type', y='MNF', ax=axs_movement[0, 1])
    # axs_movement[0, 1].set(xlabel="")
    # axs_movement[0, 1].set_title('822')
    # axs_movement[0, 1].set_ylim([0, 21])
    # sns.swarmplot(data=data_833, x='recons_type', y='MNF', ax=axs_movement[1, 1])
    # axs_movement[1, 1].set(xlabel="")
    # axs_movement[1, 1].set_title('833')
    # axs_movement[1, 1].set_ylim([0, 21])
    #
    #
    # # Create dataset for groups of movement
    # data_44_822 = data.loc[data['trials'].isin(['44_1', '44_2', '44_3', '44_4', '822', '822_contact_1'])]
    # data_821_833 = data.loc[data['trials'].isin(['821_contact_1', '821_contact_2', '821_contact_3',
    #                                              '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5',
    #                                              '821_822_2', '821_822_3',
    #                                              '833_1', '833_2', '833_3', '833_4', '833_5'])]
    #
    # # Draw swarmplot
    # fig_movement, axs_movement = pyplot.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # sns.swarmplot(data=data_44_822, x='recons_type', y='MNF', ax=axs_movement[0])
    # axs_movement[0].set(xlabel="")
    # axs_movement[0].set_title('44 and 822')
    # axs_movement[0].set_ylim([0, 21])
    # sns.swarmplot(data=data_821_833, x='recons_type', y='MNF', ax=axs_movement[1])
    # axs_movement[1].set(xlabel="")
    # axs_movement[1].set_title('821 and 833')
    # axs_movement[1].set_ylim([0, 21])
    #
    # # Create dataset by regularization term
    # data_torq_diff = data.loc[(data['subjects'].isin(['DoCi']) & data['trials'].isin(['44_1'])) |
    #                           (data['subjects'].isin(['BeLa']) & data['trials'].isin(['44_2'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_contact_2'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_3'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_4'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_822_2']))]
    # data_no_torq_diff = data.loc[(~data['subjects'].isin(['DoCi']) & data['trials'].isin(['44_1'])) |
    #                              ~(data['subjects'].isin(['BeLa']) & data['trials'].isin(['44_2'])) |
    #                              ~(data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_contact_2'])) |
    #                              ~(data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_3'])) |
    #                              ~(data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_4'])) |
    #                              ~(data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_822_2']))]
    #
    # # Draw swarmplot
    # fig_movement, axs_movement = pyplot.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # sns.swarmplot(data=data_torq_diff, x='recons_type', y='MNF', ax=axs_movement[0])
    # axs_movement[0].set(xlabel="")
    # axs_movement[0].set_title('Torque derivative')
    # axs_movement[0].set_ylim([0, 21])
    # sns.swarmplot(data=data_no_torq_diff, x='recons_type', y='MNF', ax=axs_movement[1])
    # axs_movement[1].set(xlabel="")
    # axs_movement[1].set_title('Without torque derivative')
    # axs_movement[1].set_ylim([0, 21])
    #
    # # Create dataset by regularization term pair
    # data_torq_diff = data.loc[(data['subjects'].isin(['DoCi']) & data['trials'].isin(['44_1'])) |
    #                           (data['subjects'].isin(['BeLa']) & data['trials'].isin(['44_2'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_contact_2'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_3'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_4'])) |
    #                           (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_822_2']))]
    # data_no_torq_diff = data.loc[(data['subjects'].isin(['DoCi']) & data['trials'].isin(['44_2'])) |
    #                              (data['subjects'].isin(['DoCi']) & data['trials'].isin(['44_3'])) |
    #                              (data['subjects'].isin(['BeLa']) & data['trials'].isin(['44_1'])) |
    #                              (data['subjects'].isin(['BeLa']) & data['trials'].isin(['44_3'])) |
    #                              (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_contact_1'])) |
    #                              (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_contact_3'])) |
    #                              (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_1'])) |
    #                              (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_2'])) |
    #                              (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_seul_5'])) |
    #                              (data['subjects'].isin(['SaMi']) & data['trials'].isin(['821_822_3']))]
    #
    # # Draw swarmplot
    # fig_movement, axs_movement = pyplot.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # sns.swarmplot(data=data_torq_diff, x='recons_type', y='MNF', ax=axs_movement[0])
    # axs_movement[0].set(xlabel="")
    # axs_movement[0].set_title('Torque derivative')
    # axs_movement[0].set_ylim([0, 21])
    # sns.swarmplot(data=data_no_torq_diff, x='recons_type', y='MNF', ax=axs_movement[1])
    # axs_movement[1].set(xlabel="")
    # axs_movement[1].set_title('Without torque derivative')
    # axs_movement[1].set_ylim([0, 21])

    save_path = 'Solutions/'
    fig.tight_layout
    save_name = save_path + 'MNF_controls' + '.png'
    fig.savefig(save_name)

    pyplot.show()

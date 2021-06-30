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

    diff_Q_EKF_OGE_44_all_noOGE = []
    diff_Q_EKF_OGE_821_all_noOGE = []
    diff_Q_EKF_OGE_822_all_noOGE = []
    diff_Q_EKF_OGE_833_all_noOGE = []

    diff_Q_EKF_OE_44_all_noOGE = []
    diff_Q_EKF_OE_821_all_noOGE = []
    diff_Q_EKF_OE_822_all_noOGE = []
    diff_Q_EKF_OE_833_all_noOGE = []

    diff_Q_EKF_OGE_44_all = []
    diff_Q_EKF_OGE_821_all = []
    diff_Q_EKF_OGE_822_all = []
    diff_Q_EKF_OGE_833_all = []

    diff_Q_EKF_OE_44_all = []
    diff_Q_EKF_OE_821_all = []
    diff_Q_EKF_OE_822_all = []
    diff_Q_EKF_OE_833_all = []

    diff_markers_OGE_44_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_44_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_821_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_821_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_822_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_822_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_833_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OGE_833_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}

    diff_markers_OE_44_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_44_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_821_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_821_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_822_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_822_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_833_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}
    diff_markers_OE_833_sd_all_noOGE = {'trunk': [], 'arms': [], 'legs': []}

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

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        if '44' in trial:
            diff_Q_EKF_OGE_trial_all_noOGE = diff_Q_EKF_OGE_44_all_noOGE
            diff_Q_EKF_OE_trial_all_noOGE = diff_Q_EKF_OE_44_all_noOGE
            diff_markers_OGE_trial_all_noOGE = diff_markers_OGE_44_all_noOGE
            diff_markers_OGE_trial_sd_all_noOGE = diff_markers_OGE_44_sd_all_noOGE
            diff_markers_OE_trial_all_noOGE = diff_markers_OE_44_all_noOGE
            diff_markers_OE_trial_sd_all_noOGE = diff_markers_OE_44_sd_all_noOGE
        elif '821' in trial:
            diff_Q_EKF_OGE_trial_all_noOGE = diff_Q_EKF_OGE_821_all_noOGE
            diff_Q_EKF_OE_trial_all_noOGE = diff_Q_EKF_OE_821_all_noOGE
            diff_markers_OGE_trial_all_noOGE = diff_markers_OGE_821_all_noOGE
            diff_markers_OGE_trial_sd_all_noOGE = diff_markers_OGE_821_sd_all_noOGE
            diff_markers_OE_trial_all_noOGE = diff_markers_OE_821_all_noOGE
            diff_markers_OE_trial_sd_all_noOGE = diff_markers_OE_821_sd_all_noOGE
        elif '822' in trial and '821' not in trial:
            diff_Q_EKF_OGE_trial_all_noOGE = diff_Q_EKF_OGE_822_all_noOGE
            diff_Q_EKF_OE_trial_all_noOGE = diff_Q_EKF_OE_822_all_noOGE
            diff_markers_OGE_trial_all_noOGE = diff_markers_OGE_822_all_noOGE
            diff_markers_OGE_trial_sd_all_noOGE = diff_markers_OGE_822_sd_all_noOGE
            diff_markers_OE_trial_all_noOGE = diff_markers_OE_822_all_noOGE
            diff_markers_OE_trial_sd_all_noOGE = diff_markers_OE_822_sd_all_noOGE
        elif '833' in trial:
            diff_Q_EKF_OGE_trial_all_noOGE = diff_Q_EKF_OGE_833_all_noOGE
            diff_Q_EKF_OE_trial_all_noOGE = diff_Q_EKF_OE_833_all_noOGE
            diff_markers_OGE_trial_all_noOGE = diff_markers_OGE_833_all_noOGE
            diff_markers_OGE_trial_sd_all_noOGE = diff_markers_OGE_833_sd_all_noOGE
            diff_markers_OE_trial_all_noOGE = diff_markers_OE_833_all_noOGE
            diff_markers_OE_trial_sd_all_noOGE = diff_markers_OE_833_sd_all_noOGE
        else:
            print('Unforseen trial')

        diff_Q_EKF_OGE_trial_all_noOGE.append(data['diff_Q_EKF_OGE'])
        diff_Q_EKF_OE_trial_all_noOGE.append(data['diff_Q_EKF_OE'])

        diff_markers_OGE_trial_all_noOGE['trunk'].append(data['average_distance_between_markers_OGE_trunk'])
        diff_markers_OGE_trial_sd_all_noOGE['trunk'].append(data['sd_distance_between_markers_OGE_trunk'])
        diff_markers_OGE_trial_all_noOGE['arms'].append(data['average_distance_between_markers_OGE_arms'])
        diff_markers_OGE_trial_sd_all_noOGE['arms'].append(data['sd_distance_between_markers_OGE_arms'])
        diff_markers_OGE_trial_all_noOGE['legs'].append(data['average_distance_between_markers_OGE_legs'])
        diff_markers_OGE_trial_sd_all_noOGE['legs'].append(data['sd_distance_between_markers_OGE_legs'])
        diff_markers_OE_trial_all_noOGE['trunk'].append(data['average_distance_between_markers_OE_trunk'])
        diff_markers_OE_trial_sd_all_noOGE['trunk'].append(data['sd_distance_between_markers_OE_trunk'])
        diff_markers_OE_trial_all_noOGE['arms'].append(data['average_distance_between_markers_OE_arms'])
        diff_markers_OE_trial_sd_all_noOGE['arms'].append(data['sd_distance_between_markers_OE_arms'])
        diff_markers_OE_trial_all_noOGE['legs'].append(data['average_distance_between_markers_OE_legs'])
        diff_markers_OE_trial_sd_all_noOGE['legs'].append(data['sd_distance_between_markers_OE_legs'])

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
    movement = ['44/', '821<', '822/', '833/']

    # Create dataset
    nb_44 = len(diff_Q_EKF_OE_44_all)
    nb_821 = len(diff_Q_EKF_OE_821_all)
    nb_822 = len(diff_Q_EKF_OE_822_all)
    nb_833 = len(diff_Q_EKF_OE_833_all)
    diff_Q_all = np.array(diff_Q_EKF_OGE_44_all_noOGE + diff_Q_EKF_OGE_821_all_noOGE + diff_Q_EKF_OGE_822_all_noOGE + diff_Q_EKF_OGE_833_all_noOGE +
                          diff_Q_EKF_OGE_44_all + diff_Q_EKF_OGE_821_all + diff_Q_EKF_OGE_822_all + diff_Q_EKF_OGE_833_all +
                          diff_Q_EKF_OE_44_all_noOGE + diff_Q_EKF_OE_821_all_noOGE + diff_Q_EKF_OE_822_all_noOGE + diff_Q_EKF_OE_833_all_noOGE +
                          diff_Q_EKF_OE_44_all + diff_Q_EKF_OE_821_all + diff_Q_EKF_OE_822_all + diff_Q_EKF_OE_833_all
                          )
    recons_type_Q_all = np.array(['EKF vs. Joint angle tracking,\nnormal gravity'] * 26 +
                                 ['EKF vs. Joint angle tracking,\noptimized gravity'] * 26 +
                                 ['EKF vs. Marker tracking,\nnormal gravity'] * 26 +
                                 ['EKF vs. Marker tracking,\noptimized gravity'] * 26)
    movements_Q_all = np.tile([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833, 4)

    data_Q_all = pd.DataFrame(data=diff_Q_all, index=range(diff_Q_all.size), columns=['diff_Q_all'])
    data_Q_all['recons_type'] = recons_type_Q_all
    data_Q_all['movements'] = movements_Q_all

    nb_44 = len(list(diff_markers_EKF_44_all.values())[0])
    nb_821 = len(list(diff_markers_EKF_821_all.values())[0])
    nb_822 = len(list(diff_markers_EKF_822_all.values())[0])
    nb_833 = len(list(diff_markers_EKF_833_all.values())[0])
    diff_markers_all = np.array(np.array(list(diff_markers_EKF_44_all.values()) + list(diff_markers_EKF_821_all.values()) + list(diff_markers_EKF_822_all.values()) + list(diff_markers_EKF_833_all.values()) +
                        list(diff_markers_OGE_44_all.values()) + list(diff_markers_OGE_821_all.values()) + list(diff_markers_OGE_822_all.values()) + list(diff_markers_OGE_833_all.values()) +
                        list(diff_markers_OE_44_all.values()) + list(diff_markers_OE_821_all.values()) + list(diff_markers_OE_822_all.values()) + list(diff_markers_OE_833_all.values()), dtype=object).sum())
    recons_type_marker_all = np.array(['EKF'] * 3 * 26 + ['OGE'] * 3 * 26 + ['OE'] * 3 * 26)
    movements_marker_all = np.tile(np.repeat([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833, 3), 3)
    segments_marker_all = np.array((list(np.repeat(segment, nb_44)) + list(np.repeat(segment, nb_821)) + list(np.repeat(segment, nb_822)) + list(np.repeat(segment, nb_833))) * 3)

    recons_ranges = [range(0, 26*3), range(26*3, 26*3*2), range(26*3*2, 26*3*3)]
    data_marker_all = []
    for idx, recons_range in enumerate(recons_ranges):
        data_marker_all.append(pd.DataFrame(data=diff_markers_all[recons_range], index=range(diff_markers_all[recons_range].size), columns=['diff_markers_all']))
        data_marker_all[idx]['recons_type'] = recons_type_marker_all[recons_range]
        data_marker_all[idx]['movements'] = movements_marker_all[recons_range]
        data_marker_all[idx]['segments'] = segments_marker_all[recons_range]


    def grouped_barplot(df, cat, subcat, val, ax_df):
        h_bar = []
        h_scatter = []
        u = df[cat].unique()
        x = np.arange(len(u))
        subx = df[subcat].unique()
        offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
        width = np.diff(offsets).mean()*0.9
        for i, gr in enumerate(subx):
            dfg = df[df[subcat] == gr]
            h_bar.append(ax_df.bar(x + offsets[i], dfg[val].mean(), width=width, label="{} {}".format(subcat, gr), yerr=dfg[val].std(), color=clrs_bright[i], zorder=0))
            for x_i in x:
                h_scatter.append(ax_df.scatter([x_i + offsets[i]]*len(dfg[val].values[x_i]), dfg[val].values[x_i], color='black', zorder=1))
        ax_df.set_xlabel(df['recons_type'].unique().squeeze(), fontsize=14)
        ax_df.set_xticks(x)
        ax_df.set_xticklabels(u)
        ax_df.tick_params(axis='both', labelsize=12)
        return h_bar, h_scatter

    # fig_marker_custom, axs_marker_custom = pyplot.subplots(1, 3, figsize=(20, 10), squeeze=False)
    # df = data_marker_all
    # cat = "movements"
    # subcat = "segments"
    # val = "diff_markers_all"
    # h_group_barplot = []
    # for i in range(3):
    #     h_group_barplot.append(grouped_barplot(df[i], cat, subcat, val, axs_marker_custom[0, i]))
    # axs_marker_custom[0, -1].legend(h_group_barplot[-1][0], ['Trunk', 'Arms', 'Legs'], fontsize=14)
    # axs_marker_custom[0, 0].set_ylabel('Marker error (mm)', fontsize=14)

    # Draw barplot
    # Joint angle Q difference
    fig_Q_all = pyplot.figure(figsize=(20, 10))
    axs_Q_all = fig_Q_all.gca()
    g_Q_all = sns.barplot(data=data_Q_all, x="movements", y="diff_Q_all", hue="recons_type", palette="bright", alpha=1, ax=axs_Q_all)
    g_Q_all.set_xlabel('Movements', fontsize=14)
    g_Q_all.set_ylabel('Joint angle difference (°)', fontsize=14)
    g_Q_all.tick_params(labelsize=12)
    clrs_bright = sns.color_palette("bright", 2)
    clrs_hatches_EKF = np.repeat([(clrs_bright[0], ''), (clrs_bright[0], 'x'), (clrs_bright[1], ''), (clrs_bright[1], 'x')], len(movement), axis=0)
    for i, thisbar in enumerate(g_Q_all.patches):
        thisbar.set_color(clrs_hatches_EKF[i][0])
        thisbar.set_edgecolor("white")
        thisbar.set_hatch(clrs_hatches_EKF[i][1])
    g_Q_all.legend(title="", fontsize=14)

    # Marker error difference
    # plot_ylimits = []
    # for axs in axs_marker_custom.squeeze():
    #     plot_ylimits.append(axs.get_ylim()[1])
    #
    # plot_ymax = max(plot_ylimits)
    # pyplot.setp(axs_marker_custom, ylim=(0, plot_ymax))

    print(data_Q_all.groupby(['movements', 'recons_type']).mean())

    save_path = 'Solutions/'
    fig_Q_all.tight_layout
    save_name = save_path + 'Joint_angle_diff_barplot_all_OGE_and_noOGE' + '.png'
    fig_Q_all.savefig(save_name, bbox_inches='tight')

    # fig_marker_custom.tight_layout
    # save_name = save_path + 'Marker_error_barplot_custom_OGE_and_noOGE' + '.png'
    # fig_marker_custom.savefig(save_name, bbox_inches='tight')

    pyplot.show()

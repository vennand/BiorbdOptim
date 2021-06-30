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
    # subjects_trials = [('DoCi', '822', 90), ('DoCi', '44_1', 83), ('DoCi', '44_2', 83), ('DoCi', '44_3', 83),
    #                    ('BeLa', '44_1', 83), ('BeLa', '44_2', 66), ('BeLa', '44_3', 83),
    #                    ('GuSe', '44_2', 66), ('GuSe', '44_3', 83), ('GuSe', '44_4', 83),
    #                    ('SaMi', '821_contact_1', 103), ('SaMi', '821_contact_2', 103), ('SaMi', '821_contact_3', 103), ('SaMi', '822_contact_1', 100),
    #                    ('SaMi', '821_seul_1', 100), ('SaMi', '821_seul_2', 103), ('SaMi', '821_seul_3', 103), ('SaMi', '821_seul_4', 103), ('SaMi', '821_seul_5', 103),
    #                    ('SaMi', '821_822_2', 106), ('SaMi', '821_822_3', 100),
    #                    ('JeCh', '833_1', 96), ('JeCh', '833_2', 100), ('JeCh', '833_3', 103), ('JeCh', '833_4', 100), ('JeCh', '833_5', 103),
    #                   ]

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

    diff_markers_EKF_44_segments = []
    diff_markers_EKF_821_segments = []
    diff_markers_EKF_822_segments = []
    diff_markers_EKF_833_segments = []

    diff_markers_OGE_44_segments = []
    diff_markers_OGE_821_segments = []
    diff_markers_OGE_822_segments = []
    diff_markers_OGE_833_segments = []

    diff_markers_OE_44_segments = []
    diff_markers_OE_821_segments = []
    diff_markers_OE_822_segments = []
    diff_markers_OE_833_segments = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
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
            diff_markers_EKF_trial_segments = diff_markers_EKF_44_segments
            diff_markers_OGE_trial_segments = diff_markers_OGE_44_segments
            diff_markers_OE_trial_segments = diff_markers_OE_44_segments
        elif '821' in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_821_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_821_all
            diff_markers_EKF_trial_all = diff_markers_EKF_821_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_821_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_821_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_821_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_821_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_821_sd_all
            diff_markers_EKF_trial_segments = diff_markers_EKF_821_segments
            diff_markers_OGE_trial_segments = diff_markers_OGE_821_segments
            diff_markers_OE_trial_segments = diff_markers_OE_821_segments
        elif '822' in trial and '821' not in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_822_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_822_all
            diff_markers_EKF_trial_all = diff_markers_EKF_822_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_822_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_822_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_822_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_822_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_822_sd_all
            diff_markers_EKF_trial_segments = diff_markers_EKF_822_segments
            diff_markers_OGE_trial_segments = diff_markers_OGE_822_segments
            diff_markers_OE_trial_segments = diff_markers_OE_822_segments
        elif '833' in trial:
            diff_Q_EKF_OGE_trial_all = diff_Q_EKF_OGE_833_all
            diff_Q_EKF_OE_trial_all = diff_Q_EKF_OE_833_all
            diff_markers_EKF_trial_all = diff_markers_EKF_833_all
            diff_markers_EKF_trial_sd_all = diff_markers_EKF_833_sd_all
            diff_markers_OGE_trial_all = diff_markers_OGE_833_all
            diff_markers_OGE_trial_sd_all = diff_markers_OGE_833_sd_all
            diff_markers_OE_trial_all = diff_markers_OE_833_all
            diff_markers_OE_trial_sd_all = diff_markers_OE_833_sd_all
            diff_markers_EKF_trial_segments = diff_markers_EKF_833_segments
            diff_markers_OGE_trial_segments = diff_markers_OGE_833_segments
            diff_markers_OE_trial_segments = diff_markers_OE_833_segments
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
        diff_markers_EKF_trial_segments.append(data['segment_marker_error_EKF_biorbd'])
        diff_markers_OGE_trial_segments.append(data['segment_marker_error_OGE'])
        diff_markers_OE_trial_segments.append(data['segment_marker_error_OE'])

    segment = ['Trunk', 'Arms', 'Legs']
    movement = ['44/', '821<', '822/', '833/']

    nb_44 = len(diff_Q_EKF_OE_44_all)
    nb_821 = len(diff_Q_EKF_OE_821_all)
    nb_822 = len(diff_Q_EKF_OE_822_all)
    nb_833 = len(diff_Q_EKF_OE_833_all)
    diff_Q_all = np.array(diff_Q_EKF_OGE_44_all + diff_Q_EKF_OGE_821_all + diff_Q_EKF_OGE_822_all + diff_Q_EKF_OGE_833_all +
                        diff_Q_EKF_OE_44_all + diff_Q_EKF_OE_821_all + diff_Q_EKF_OE_822_all + diff_Q_EKF_OE_833_all)
    recons_type_Q_all = np.array(['EKF vs. OGE'] * 26 + ['EKF vs. OE'] * 26)
    movements_Q_all = np.tile([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833, 2)

    data_Q_all = pd.DataFrame(data=diff_Q_all, index=range(diff_Q_all.size), columns=['diff_Q_all'])
    data_Q_all['std_all'] = None
    data_Q_all['recons_type'] = recons_type_Q_all
    data_Q_all['movements'] = movements_Q_all

    nb_44 = len(list(diff_markers_EKF_44_all.values())[0])
    nb_821 = len(list(diff_markers_EKF_821_all.values())[0])
    nb_822 = len(list(diff_markers_EKF_822_all.values())[0])
    nb_833 = len(list(diff_markers_EKF_833_all.values())[0])
    diff_markers_all = np.array(list(diff_markers_EKF_44_all.values()) + list(diff_markers_EKF_821_all.values()) + list(diff_markers_EKF_822_all.values()) + list(diff_markers_EKF_833_all.values()) +
                        list(diff_markers_OGE_44_all.values()) + list(diff_markers_OGE_821_all.values()) + list(diff_markers_OGE_822_all.values()) + list(diff_markers_OGE_833_all.values()) +
                        list(diff_markers_OE_44_all.values()) + list(diff_markers_OE_821_all.values()) + list(diff_markers_OE_822_all.values()) + list(diff_markers_OE_833_all.values()), dtype=object)
    diff_markers_segments = np.repeat(np.array([diff_markers_EKF_44_segments, diff_markers_EKF_821_segments, diff_markers_EKF_822_segments, diff_markers_EKF_833_segments,
                                      diff_markers_OGE_44_segments, diff_markers_OGE_821_segments, diff_markers_OGE_822_segments, diff_markers_OGE_833_segments,
                                      diff_markers_OE_44_segments, diff_markers_OE_821_segments, diff_markers_OE_822_segments, diff_markers_OE_833_segments], dtype=object), 3)
    recons_type_marker = np.array(['EKF'] * 3 * 4 + ['Joint angle tracking'] * 3 * 4 + ['Marker tracking'] * 3 * 4)
    movements_marker = np.array(list(np.repeat(movement, 3)) * 3)
    segments_marker = np.array(segment * 4 * 3)

    recons_ranges = [range(0, 12), range(12, 24), range(24, 36)]
    data_marker = []
    for idx, recons_range in enumerate(recons_ranges):
        data_marker.append(pd.DataFrame(data=diff_markers_all[recons_range], index=range(diff_markers_all[recons_range].size), columns=['diff_markers_all']))
        data_marker[idx]['diff_markers_segments'] = diff_markers_segments[recons_range]
        data_marker[idx]['recons_type'] = recons_type_marker[recons_range]
        data_marker[idx]['movements'] = movements_marker[recons_range]
        data_marker[idx]['segments'] = segments_marker[recons_range]

    clrs_bright = sns.color_palette("bright", 3)
    clrs_dark = sns.color_palette("dark", 3)

    def grouped_barplot(df, cat, subcat, val, seg, ax_df):
        h_bar = []
        h_scatter = []
        h_scatter_seg = []
        u = df[cat].unique()
        x = np.arange(len(u))
        subx = df[subcat].unique()
        offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
        width = np.diff(offsets).mean()*0.9
        for i, gr in enumerate(subx):
            dfg = df[df[subcat] == gr]
            h_bar.append(ax_df.bar(x + offsets[i], dfg[val].apply(lambda x: np.mean(x)).values , width=width, label="{} {}".format(subcat, gr), yerr=dfg[val].apply(lambda x: np.std(x)).values, color=clrs_bright[i], zorder=0))
            for x_i in x:
                h_scatter.append(ax_df.scatter([x_i + offsets[i]]*len(dfg[val].values[x_i]), dfg[val].values[x_i], color='black', zorder=1))
                # if i==1:
                #     for seg_j in dfg[seg].values[x_i]:
                #         h_scatter_seg.append(ax_df.scatter(x_i + offsets[i], seg_j['MainD'], color='purple', zorder=1))
                #         h_scatter_seg.append(ax_df.scatter(x_i + offsets[i], seg_j['MainG'], color='red', zorder=1))
                # if i==2:
                #     for seg_j in dfg[seg].values[x_i]:
                #         h_scatter_seg.append(ax_df.scatter(x_i + offsets[i], seg_j['PiedD'], color='purple', zorder=1))
                #         h_scatter_seg.append(ax_df.scatter(x_i + offsets[i], seg_j['PiedG'], color='red', zorder=1))
        ax_df.set_xlabel(df['recons_type'].unique().squeeze(), fontsize=14)
        ax_df.set_xticks(x)
        ax_df.set_xticklabels(u)
        ax_df.tick_params(axis='both', labelsize=12)
        return h_bar, h_scatter, h_scatter_seg

    fig_marker_custom, axs_marker_custom = pyplot.subplots(1, 3, figsize=(20, 10), squeeze=False)
    df = data_marker
    cat = "movements"
    subcat = "segments"
    val = "diff_markers_all"
    seg = 'diff_markers_segments'
    h_group_barplot = []
    for i in range(3):
        h_group_barplot.append(grouped_barplot(df[i], cat, subcat, val, seg, axs_marker_custom[0, i]))
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

    plot_ylimits = []
    for axs in axs_marker_custom.squeeze():
        plot_ylimits.append(axs.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    # plot_ymax = 360
    pyplot.setp(axs_marker_custom, ylim=(0, plot_ymax))

    print(data_Q_all.groupby(['movements', 'recons_type']).mean())
    print(data_Q_all.groupby(['movements', 'recons_type']).std())
    print('EKF marker error: ', np.mean(data_marker[0]['diff_markers_all'].sum()), ' ± ', np.std(data_marker[0]['diff_markers_all'].sum()))
    print('OGE marker error: ', np.mean(data_marker[1]['diff_markers_all'].sum()), ' ± ', np.std(data_marker[1]['diff_markers_all'].sum()))
    print('OE marker error: ', np.mean(data_marker[2]['diff_markers_all'].sum()), ' ± ', np.std(data_marker[2]['diff_markers_all'].sum()))

    # End chain mean RMS
    end_chain_RMS_hands = []
    end_chain_RMS_feet = []
    for idx, method in enumerate(data_marker):
        end_chain_RMS_hands.append([])
        end_chain_RMS_feet.append([])
        segments_marker_trial = [item for sublist in method[method['segments'] == 'Trunk']['diff_markers_segments'].to_list() for item in sublist]
        for trial in segments_marker_trial:
            end_chain_RMS_hands[idx].append([trial['MainD'], trial['MainG']])
            end_chain_RMS_feet[idx].append([trial['PiedD'], trial['PiedG']])

    end_chain_821_RMS_hands = []
    end_chain_833_RMS_hands = []
    for idx, method in enumerate(data_marker):
        end_chain_821_RMS_hands.append([])
        segments_marker_trial_821 = [item for sublist in method[(method['segments'] == 'Trunk') & (method['movements'] == '821<')]['diff_markers_segments'].to_list() for item in sublist]
        for trial in segments_marker_trial_821:
            end_chain_821_RMS_hands[idx].append([trial['MainD']])

        end_chain_833_RMS_hands.append([])
        segments_marker_trial_833 = [item for sublist in method[(method['segments'] == 'Trunk') & (method['movements'] == '833/')]['diff_markers_segments'].to_list() for item in sublist]
        for trial in segments_marker_trial_833:
            end_chain_833_RMS_hands[idx].append([trial['MainG']])

    print('Hands mean RMS: ', [np.nanmean(method) for method in end_chain_RMS_hands])
    print('Feet mean RMS: ', [np.nanmean(method) for method in end_chain_RMS_feet])
    print('Right hand mean RMS for 821: ', [np.nanmean(method) for method in end_chain_821_RMS_hands])
    print('Left hand mean RMS for 833: ', [np.nanmean(method) for method in end_chain_833_RMS_hands])

    save_path = 'Solutions/'
    fig_Q_all.tight_layout
    save_name = save_path + 'Joint_angle_diff_barplot_all_noOGE' + '.png'
    fig_Q_all.savefig(save_name, bbox_inches='tight')

    fig_marker_custom.tight_layout
    save_name = save_path + 'Marker_error_barplot_custom_noOGE' + '.png'
    fig_marker_custom.savefig(save_name, bbox_inches='tight')

    pyplot.show()

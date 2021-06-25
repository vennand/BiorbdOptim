import biorbd
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

    controls_OE = []

    states_EKF = []
    controls_EKF = []
    momentum_EKF = []
    RMSE_momentum_EKF = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        controls_OE.append(data['controls_OE'])

        states_EKF.append(data['states_EKF'])
        controls_EKF.append(data['controls_EKF'])
        momentum_EKF.append(data['momentum_EKF'])
        RMSE_momentum_EKF.append(data['RMSE_momentum_EKF'])

    peak_to_peak_momentum_EKF = [np.max(trial, axis=1) - np.min(trial, axis=1) for trial in momentum_EKF]
    peak_to_peak_momentum_EKF_max = np.max(peak_to_peak_momentum_EKF)
    peak_to_peak_momentum_EKF_mean = np.mean(peak_to_peak_momentum_EKF)
    peak_to_peak_momentum_EKF_std = np.std(peak_to_peak_momentum_EKF)
    RMSE_momentum_EKF_mean = np.mean(RMSE_momentum_EKF)
    RMSE_momentum_EKF_std = np.mean(RMSE_momentum_EKF)

    abs_max_controls_root_EKF = np.max([np.max(np.abs(trial['tau'].full())) for trial in controls_EKF])

    # RMS_controls = np.sqrt(np.mean(np.concatenate([np.ravel(trial_EKF['tau'].full()[6:, :] - trial_OE['tau'][6:, :]) ** 2 for trial_EKF, trial_OE in zip(controls_EKF, controls_OE)])))
    RMS_controls = [np.sqrt(np.mean((trial_EKF['tau'].full()[6:, :] - trial_OE['tau'][6:, :]) ** 2)) for trial_EKF, trial_OE in zip(controls_EKF, controls_OE)]
    RMS_controls_mean = np.mean(RMS_controls)
    RMS_controls_std = np.std(RMS_controls)

    # In EKF, angular momentum was not constant with peak - to - peak variations up to XXX kgm / s.
    # This resulted in residual torques up to XX N XX Nm on the root segment.
    # The root mean square difference in joint torques between the two methods was XXXX Nm.

    # Create dataset
    diff_Q = np.array(diff_Q_EKF_OE_44 + diff_Q_EKF_OE_821 + diff_Q_EKF_OE_822 + diff_Q_EKF_OE_833)
    recons_type_Q = np.array(['EKF vs. OE'] * 3 * 4)
    movements_Q = np.array(list(np.repeat(movement, 3)))
    segments_Q = np.array(segment * 4)

    # recons_ranges = [range(0, 12), range(12, 24)]
    recons_ranges = [range(0, 12)]
    data_Q = []
    for idx, recons_range in enumerate(recons_ranges):
        data_Q.append(pd.DataFrame(data=diff_Q[recons_range], index=range(diff_Q[recons_range].size), columns=['diff_Q']))
        data_Q[idx]['recons_type'] = recons_type_Q[recons_range]
        data_Q[idx]['movements'] = movements_Q[recons_range]
        data_Q[idx]['segments'] = segments_Q[recons_range]

    nb_44 = len(diff_Q_EKF_OE_44_all)
    nb_821 = len(diff_Q_EKF_OE_821_all)
    nb_822 = len(diff_Q_EKF_OE_822_all)
    nb_833 = len(diff_Q_EKF_OE_833_all)
    diff_Q_all = np.array(diff_Q_EKF_OE_44_all + diff_Q_EKF_OE_821_all + diff_Q_EKF_OE_822_all + diff_Q_EKF_OE_833_all)
    recons_type_Q_all = np.array(['EKF vs. OE'] * 26)
    movements_Q_all = np.array([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833)

    data_Q_all = pd.DataFrame(data=diff_Q_all, index=range(diff_Q_all.size), columns=['diff_Q_all'])
    data_Q_all['std_all'] = None
    data_Q_all['recons_type'] = recons_type_Q_all
    data_Q_all['movements'] = movements_Q_all

    diff_markers = np.array(diff_markers_EKF_44 + diff_markers_EKF_821 + diff_markers_EKF_822 + diff_markers_EKF_833 +
                            diff_markers_OE_44 + diff_markers_OE_821 + diff_markers_OE_822 + diff_markers_OE_833)
    diff_markers_sd = np.array(diff_markers_EKF_44_sd + diff_markers_EKF_821_sd + diff_markers_EKF_822_sd + diff_markers_EKF_833_sd +
                            diff_markers_OE_44_sd + diff_markers_OE_821_sd + diff_markers_OE_822_sd + diff_markers_OE_833_sd)
    diff_markers_all = np.array(list(diff_markers_EKF_44_all.values()) + list(diff_markers_EKF_821_all.values()) + list(diff_markers_EKF_822_all.values()) + list(diff_markers_EKF_833_all.values()) +
                        list(diff_markers_OE_44_all.values()) + list(diff_markers_OE_821_all.values()) + list(diff_markers_OE_822_all.values()) + list(diff_markers_OE_833_all.values()), dtype=object)
    diff_markers_sd = np.array([np.std(i) for i in diff_markers_all])
    diff_markers_sd_all = np.array(list(diff_markers_EKF_44_sd_all.values()) + list(diff_markers_EKF_821_sd_all.values()) + list(diff_markers_EKF_822_sd_all.values()) + list(diff_markers_EKF_833_sd_all.values()) +
                        list(diff_markers_OE_44_sd_all.values()) + list(diff_markers_OE_821_sd_all.values()) + list(diff_markers_OE_822_sd_all.values()) + list(diff_markers_OE_833_sd_all.values()), dtype=object)
    recons_type_marker = np.array(['EKF'] * 3 * 4 + ['OE'] * 3 * 4)
    movements_marker = np.array(list(np.repeat(movement, 3)) * 2)
    segments_marker = np.array(segment * 4 * 2)

    recons_ranges = [range(0, 12), range(12, 24)]
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
                        list(diff_markers_OE_44_all.values()) + list(diff_markers_OE_821_all.values()) + list(diff_markers_OE_822_all.values()) + list(diff_markers_OE_833_all.values()), dtype=object).sum())
    diff_markers_sd_all = np.array(np.array(list(diff_markers_EKF_44_sd_all.values()) + list(diff_markers_EKF_821_sd_all.values()) + list(diff_markers_EKF_822_sd_all.values()) + list(diff_markers_EKF_833_sd_all.values()) +
                        list(diff_markers_OE_44_sd_all.values()) + list(diff_markers_OE_821_sd_all.values()) + list(diff_markers_OE_822_sd_all.values()) + list(diff_markers_OE_833_sd_all.values()), dtype=object).sum())
    recons_type_marker_all = np.array(['EKF'] * 3 * 26 + ['OE'] * 3 * 26)
    movements_marker_all = np.tile(np.repeat([movement[0]] * nb_44 + [movement[1]] * nb_821 + [movement[2]] * nb_822 + [movement[3]] * nb_833, 3), 2)
    segments_marker_all = np.array((list(np.repeat(segment, nb_44)) + list(np.repeat(segment, nb_821)) + list(np.repeat(segment, nb_822)) + list(np.repeat(segment, nb_833))) * 2)

    recons_ranges = [range(0, 26*3), range(26*3, 26*3*2)]
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

    clrs_hatches_EKF_vs_OE = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_Q[0].loc[data_Q[0]['movements'] == y]['diff_Q']))
                               else (clrs_bright[idx], 'x')
                               for y in movement
                               for idx, x in enumerate(data_Q[0].loc[data_Q[0]['movements'] == y]['diff_Q'])]

    clrs_hatches_EKF = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_marker[0].loc[data_marker[0]['movements'] == y]['diff_markers']))
                        else (clrs_bright[idx], 'x')
                        for y in movement
                        for idx, x in enumerate(data_marker[0].loc[data_marker[0]['movements'] == y]['diff_markers'])]
    clrs_hatches_OE = [(tuple(z*alpha + (1-alpha) for z in clrs_dark[idx]), '') if (x < max(data_marker[1].loc[data_marker[1]['movements'] == y]['diff_markers']))
                       else (clrs_bright[idx], 'x')
                       for y in movement
                       for idx, x in enumerate(data_marker[1].loc[data_marker[1]['movements'] == y]['diff_markers'])]

    clrs_hatches_EKF_vs_OE[:] = [clrs_hatches_EKF_vs_OE[i] for i in patches_order]

    clrs_hatches_EKF[:] = [clrs_hatches_EKF[i] for i in patches_order]
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
        ax_df.set_xlabel(df['recons_type'].unique().squeeze(), fontsize=20)
        ax_df.set_xticks(x)
        ax_df.set_xticklabels(u)
        ax_df.tick_params(axis='both', labelsize=18)
        return h_bar, h_scatter

    fig_marker_custom, axs_marker_custom = pyplot.subplots(1, 2, figsize=(20, 10), squeeze=False)
    df = data_marker
    cat = "movements"
    subcat = "segments"
    val = "diff_markers"
    err = "std"
    all_val = "diff_markers_all"
    h_group_barplot = []
    for i in range(2):
        h_group_barplot.append(grouped_barplot(df[i], cat, subcat, val, err, all_val, axs_marker_custom[0, i]))
    axs_marker_custom[0, -1].legend(h_group_barplot[-1][0], ['Trunk', 'Arms', 'Legs'], fontsize=16)
    axs_marker_custom[0, 0].set_ylabel('Marker error (mm)', fontsize=20)

    # Draw barplot
    # Joint angle Q difference
    fig_Q_all = pyplot.figure(figsize=(20, 10))
    axs_Q_all = fig_Q_all.gca()
    g_Q_all = sns.barplot(data=data_Q_all, x="movements", y="diff_Q_all", palette="bright", alpha=1, ax=axs_Q_all)
    g_Q_all.set_xlabel('Movements', fontsize=20)
    g_Q_all.set_ylabel('Joint angle diffenrence (°)', fontsize=20)
    g_Q_all.tick_params(labelsize=18)
    # g_Q_all.legend(title="", fontsize=14)

    fig_Q = pyplot.figure(figsize=(20, 10))
    axs_Q = fig_Q.gca()
    g_EKF_vs_OE = sns.barplot(data=data_Q[0], x="movements", y="diff_Q", hue="segments", palette="bright", alpha=1, ax=axs_Q)
    g_EKF_vs_OE.set_xlabel("EKF vs. OE", fontsize=14)
    g_EKF_vs_OE.set_ylabel("")
    g_EKF_vs_OE.tick_params(labelsize=12)
    g_EKF_vs_OE.legend(title="")
    for i, thisbar in enumerate(g_EKF_vs_OE.patches):
        thisbar.set_color(clrs_hatches_EKF_vs_OE[i][0])
        thisbar.set_edgecolor("white")

    # Marker error difference
    fig_marker, axs_marker = pyplot.subplots(1, 2, figsize=(20, 10), squeeze=False)
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

    g_OE = sns.barplot(data=data_marker[1], x="movements", y="diff_markers", hue="segments", ci="std", palette="bright", alpha=1, ax=axs_marker[0, 1])
    g_OE.set_xlabel("OE", fontsize=14)
    g_OE.set_ylabel("")
    g_OE.tick_params(labelsize=12)
    g_OE.legend(title="")
    for i, thisbar in enumerate(g_OE.patches):
        thisbar.set_color(clrs_hatches_OE[i][0])
        thisbar.set_edgecolor("white")
        # thisbar.set_hatch(clrs_hatches_OE[i][1])

    plot_ylimits = []
    plot_ylimits.append(g_EKF_vs_OE.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    g_EKF_vs_OE.set(ylim=(0, plot_ymax))

    plot_ylimits = []
    plot_ylimits.append(g_EKF.get_ylim()[1])
    plot_ylimits.append(g_OE.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    g_EKF.set(ylim=(0, plot_ymax))
    g_OE.set(ylim=(0, plot_ymax))

    plot_ylimits = []
    for axs in axs_marker_custom.squeeze():
        plot_ylimits.append(axs.get_ylim()[1])

    plot_ymax = max(plot_ylimits)
    pyplot.setp(axs_marker_custom, ylim=(0, plot_ymax))

    save_path = 'Solutions/'
    fig_Q_all.tight_layout
    save_name = save_path + 'Joint_angle_diff_barplot_all_noOGE' + '.png'
    fig_Q_all.savefig(save_name, bbox_inches='tight')

    fig_Q.tight_layout
    save_name = save_path + 'Joint_angle_diff_barplot_noOGE' + '.png'
    fig_Q.savefig(save_name, bbox_inches='tight')

    fig_marker.tight_layout
    save_name = save_path + 'Marker_error_barplot_noOGE' + '.png'
    fig_marker.savefig(save_name, bbox_inches='tight')

    fig_marker_custom.tight_layout
    save_name = save_path + 'Marker_error_barplot_custom_noOGE' + '.png'
    fig_marker_custom.savefig(save_name, bbox_inches='tight')

    pyplot.show()

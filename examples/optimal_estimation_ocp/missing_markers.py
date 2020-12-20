import biorbd
import ezc3d
from casadi import MX, Function
import numpy as np
import os
import sys
import pickle
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers

from biorbd_optim import (
    OptimalControlProgram,
    Data,
)

def states_to_markers(biorbd_model, ocp, states):
    q = states['q']
    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    return markers

essai = {'DoCi': ['822', '44_1', '44_2', '44_3'],
         'JeCh': ['833_1', '833_2', '833_3', '833_4', '833_5'],
         'BeLa': ['44_1', '44_2', '44_3'],
         'GuSe': ['44_2', '44_3', '44_4'],
         'SaMi': ['821_822_2', '821_822_3',
                  '821_contact_1', '821_contact_2', '821_contact_3',
                  '822_contact_1',
                  '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5']}

nb_trials = np.sum([len(trial) for trial in essai.values()])
fig, axs = pyplot.subplots(nrows=nb_trials, ncols=1, figsize=(20, 10))
axs_counter = -1

nb_nan = dict()
for subject, trials in essai.items():
    # print('Subject: ', subject)
    nb_nan[subject] = dict()

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'
    for trial in trials:
        if (subject == 'BeLa' and trial == '44_2') or (subject == 'GuSe' and trial == '44_2'):
            number_shooting_points = 80
        else:
            number_shooting_points = 100
        data_filename = load_data_filename(subject, trial)
        model_name = data_filename['model']
        c3d_name = data_filename['c3d']
        frames = data_filename['frames']

        biorbd_model = biorbd.Model(model_path + model_name)
        c3d = ezc3d.c3d(c3d_path + c3d_name)

        frequency = c3d['header']['points']['frame_rate']
        duration = len(frames) / frequency

        # --- Adjust number of shooting points --- #
        adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

        # --- Load results --- #
        load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
        load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
        ocp, sol = OptimalControlProgram.load(load_name + ".bo")
        states, _ = Data.get_data(ocp, sol)

        load_variables_name = load_name + ".pkl"
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        markers_mocap = data['mocap']
        frames = data['frames']
        step_size = data['step_size']
        gravity = data['gravity']

        load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
        load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
        with open(load_variables_name, 'rb') as handle:
            kalman_states = pickle.load(handle)
        q_kalman_biorbd = kalman_states['q'][:, ::step_size]
        qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
        qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

        states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}

        # --- Marker error --- #
        biorbd_model.setGravity(biorbd.Vector3d(gravity[0], gravity[1], gravity[2]))
        markers = states_to_markers(biorbd_model, ocp, states)
        markers_kalman_biorbd = states_to_markers(biorbd_model, ocp, states_kalman_biorbd)

        markers_error_OE = np.nanmean(np.sqrt(np.sum((markers - markers_mocap) ** 2, axis=0)) * 1000)
        markers_error_EKF_biorbd = np.nanmean(np.sqrt(np.sum((markers_kalman_biorbd - markers_mocap) ** 2, axis=0)) * 1000)

        # --- Count missing markers --- #
        step_size = 1
        markers_all, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

        nb_nan[subject][trial] = np.count_nonzero(np.isnan(markers_all[0, :, :]), axis=0) / markers_all.shape[1]*100

        axs_counter += 1
        im = axs[axs_counter].imshow(nb_nan[subject][trial][np.newaxis, :], cmap="plasma", aspect="auto")

        axs[axs_counter].plot([0, 1.1], [0, 0], color='black', lw=1, transform=axs[axs_counter].transAxes, clip_on=False)
        axs[axs_counter].plot([0, 1.1], [1, 1], color='black', lw=1, transform=axs[axs_counter].transAxes, clip_on=False)

        if axs_counter == 0:
            axs[axs_counter].text(markers_all.shape[2] * 1.08, -0.9, 'Marker error\n(mm)', ha='right', fontsize=9)
        axs[axs_counter].text(markers_all.shape[2]*1.08, -0.05, 'OE: ' + f"{np.linalg.norm(markers_error_OE):.2f}", ha='right', fontsize=9)
        axs[axs_counter].text(markers_all.shape[2]*1.08, 0.45, 'EKF: ' + f"{np.linalg.norm(markers_error_EKF_biorbd):.2f}", ha='right', fontsize=9)

        axs[axs_counter].set_ylabel(subject + ', ' + trial, rotation=0, fontsize=9, horizontalalignment='right')
        axs[axs_counter].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        axs[axs_counter].tick_params(axis="y", direction='in', which='both', left=False, right=False, labelleft=False)

        #remove axis
        #axis labels
axs[nb_trials-1].set_xlabel('Aerial time')

cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95, pad=0.1)
cbar.ax.set_ylabel('% of missing markers', rotation=270, labelpad=15)
cbar.ax.tick_params(axis="y", direction='in', which='both')
# fig.align_ylabels()

# pyplot.tight_layout()

save_path = 'Solutions/'
save_name = save_path + 'Missing_markers_and_errors' + '.png'
pyplot.savefig(save_name)

pyplot.show()


import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function
import pickle
from scipy.io import loadmat
import os
import sys
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers

from biorbd_optim import (
    OptimalControlProgram,
    Simulate,
    Problem,
    Bounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
)

def rotating_gravity(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


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


if __name__ == "__main__":
    # subject = 'DoCi'
    subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '833_2'

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

    biorbd_model = biorbd.Model(model_path + model_name)
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Functions --- #
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)
    fd = biorbd.to_casadi_func("fd", biorbd_model.ForwardDynamics, q, qdot, tau)

    # --- Load --- #
    load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
    ocp, sol = OptimalControlProgram.load(load_name + ".bo")

    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    markers_mocap = data['mocap']
    frames = data['frames']
    step_size = data['step_size']

    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    optimal_gravity_filename = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".bo"

    q_kalman = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size]
    qdot_kalman = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
    qddot_kalman = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]

    states_kalman = {'q': q_kalman, 'q_dot': qdot_kalman}
    controls_kalman = {'tau': id(q_kalman, qdot_kalman, qddot_kalman)}

    load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)
    q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

    states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
    controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)}

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)
    qddot = fd(states['q'], states['q_dot'], controls['tau'])

    ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)
    angle = params_optimal_gravity["gravity_angle"].squeeze()
    qddot_optimal_gravity = fd(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], controls_optimal_gravity['tau'])

    rotating_gravity(biorbd_model, angle.squeeze())
    markers = states_to_markers(biorbd_model, ocp, states)
    markers_optimal_gravity = states_to_markers(biorbd_model, ocp, states_optimal_gravity)
    markers_kalman = states_to_markers(biorbd_model, ocp, states_kalman)
    markers_kalman_biorbd = states_to_markers(biorbd_model, ocp, states_kalman_biorbd)

    # --- Marker error --- #
    markers_error_OE = np.sqrt(np.sum((markers - markers_mocap)**2, axis=0))*1000
    markers_error_OGE = np.sqrt(np.sum((markers_optimal_gravity - markers_mocap)**2, axis=0))*1000
    markers_error_EKF_matlab = np.sqrt(np.sum((markers_kalman - markers_mocap)**2, axis=0))*1000
    markers_error_EKF_biorbd = np.sqrt(np.sum((markers_kalman_biorbd - markers_mocap)**2, axis=0))*1000

    # --- Missing markers --- #
    missing_markers_per_node = np.count_nonzero(np.isnan(markers_mocap[0, :, :]), axis=0) / markers_mocap.shape[1] * 100

    # --- By segments --- #
    segments = dict()
    for segment_idx in range(biorbd_model.nbSegment()):
        segments[biorbd_model.segment(segment_idx).id()] = {'name': biorbd_model.segment(segment_idx).name().to_string(),
                                                            'markers_idx': []}
    for marker_idx in range(biorbd_model.nbMarkers()):
        segments[biorbd_model.marker(marker_idx).parentId()]['markers_idx'].append(marker_idx)

    segment_names = []
    segment_marker_error_OE = []
    segment_marker_error_OGE = []
    segment_marker_error_EKF_matlab = []
    segment_marker_error_EKF_biorbd = []
    segment_marker_error_OE_per_node = []
    segment_marker_error_OGE_per_node = []
    segment_marker_error_EKF_matlab_per_node = []
    segment_marker_error_EKF_biorbd_per_node = []
    segment_missing_markers = []
    segment_missing_markers_per_node = []
    for segment in segments.values():
        segment['markers_error_OE'] = np.nanmean([markers_error_OE[marker_idx, :] for marker_idx in segment['markers_idx']])
        segment['markers_error_OGE'] = np.nanmean([markers_error_OGE[marker_idx, :] for marker_idx in segment['markers_idx']])
        segment['markers_error_EKF_matlab'] = np.nanmean([markers_error_EKF_matlab[marker_idx, :] for marker_idx in segment['markers_idx']])
        segment['markers_error_EKF_biorbd'] = np.nanmean([markers_error_EKF_biorbd[marker_idx, :] for marker_idx in segment['markers_idx']])

        segment_names.append(segment['name'])
        segment_marker_error_OE.append(segment['markers_error_OE'])
        segment_marker_error_OGE.append(segment['markers_error_OGE'])
        segment_marker_error_EKF_matlab.append(segment['markers_error_EKF_matlab'])
        segment_marker_error_EKF_biorbd.append(segment['markers_error_EKF_biorbd'])

        segment_marker_error_OE_per_node.append(np.nanmean([markers_error_OE[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0))
        segment_marker_error_OGE_per_node.append(np.nanmean([markers_error_OGE[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0))
        segment_marker_error_EKF_matlab_per_node.append(np.nanmean([markers_error_EKF_matlab[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0))
        segment_marker_error_EKF_biorbd_per_node.append(np.nanmean([markers_error_EKF_biorbd[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0))

        segment_missing_markers.append(np.count_nonzero(np.isnan([markers_mocap[0, marker_idx, :] for marker_idx in segment['markers_idx']])) / len(segment['markers_idx']) / markers_mocap.shape[2] * 100)
        segment_missing_markers_per_node.append(np.count_nonzero(np.isnan([markers_mocap[0, marker_idx, :] for marker_idx in segment['markers_idx']]), axis=0) / len(segment['markers_idx']) * 100)

    print('Segment names')
    print(*segment_names, sep=',')
    print('Marker error OE, per segment')
    print(*segment_marker_error_OE, sep=',')
    print('Marker error OGE, per segment')
    print(*segment_marker_error_OGE, sep=',')
    print('Marker error EKF matlab, per segment')
    print(*segment_marker_error_EKF_matlab, sep=',')
    print('Marker error EKF biorbd, per segment')
    print(*segment_marker_error_EKF_biorbd, sep=',')

    print('Missing markers, per segment')
    print(*segment_missing_markers, sep=',')

    print('Marker error OE, per node')
    print(*np.nanmean(markers_error_OE, axis=0), sep=',')
    print('Marker error OGE, per node')
    print(*np.nanmean(markers_error_OGE, axis=0), sep=',')
    print('Marker error EKF matlab, per node')
    print(*np.nanmean(markers_error_EKF_matlab, axis=0), sep=',')
    print('Marker error EKF biorbd, per node')
    print(*np.nanmean(markers_error_EKF_biorbd, axis=0), sep=',')

    print('Missing markers, per node')
    print(*missing_markers_per_node, sep=',')

    print('Marker error OE, per segment, per node')
    for segment in segment_marker_error_OE_per_node:
        print(*segment, sep=',')
    print('Marker error OGE, per segment, per node')
    for segment in segment_marker_error_OGE_per_node:
        print(*segment, sep=',')
    print('Marker error EKF matlab, per segment, per node')
    for segment in segment_marker_error_EKF_matlab_per_node:
        print(*segment, sep=',')
    print('Marker error EKF biorbd, per segment, per node')
    for segment in segment_marker_error_EKF_biorbd_per_node:
        print(*segment, sep=',')

    print('Missing markers, per segment, per node')
    for segment in segment_missing_markers_per_node:
        print(*segment, sep=',')

    # --- Stats --- #
    #OE
    average_distance_between_markers = np.nanmean(markers_error_OE)
    sd_distance_between_markers = np.nanstd(markers_error_OE)

    # OG
    average_distance_between_markers_optimal_gravity = np.nanmean(markers_error_OGE)
    sd_distance_between_markers_optimal_gravity = np.nanstd(markers_error_OGE)

    # EKF
    average_distance_between_markers_kalman = np.nanmean(markers_error_EKF_matlab)
    sd_distance_between_markers_kalman = np.nanstd(markers_error_EKF_matlab)

    # EKF biorbd
    average_distance_between_markers_kalman_biorbd = np.nanmean(markers_error_EKF_biorbd)
    sd_distance_between_markers_kalman_biorbd = np.nanstd(markers_error_EKF_biorbd)

    # print('Number of shooting points: ', adjusted_number_shooting_points)
    # print('Average marker error')
    # print('Kalman: ', average_distance_between_markers_kalman, u"\u00B1", sd_distance_between_markers_kalman)
    # print('Kalman biorbd: ', average_distance_between_markers_kalman_biorbd, u"\u00B1", sd_distance_between_markers_kalman_biorbd)
    # print('Optimal gravity: ', average_distance_between_markers_optimal_gravity, u"\u00B1", sd_distance_between_markers_optimal_gravity)
    # print('Estimation: ', average_distance_between_markers, u"\u00B1", sd_distance_between_markers)

    # --- Plots --- #
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
    dofs = [range(0, 6)]
    # for idx_dof, dof in enumerate(dofs):
    #     fig, axs = pyplot.subplots(2)
    #     axs[0].plot(states_kalman['q'][dof, :].T, color='blue')
    #     axs[0].plot(states_optimal_gravity['q'][dof, :].T, color='red')
    #     axs[0].plot(states['q'][dof, :].T, color='green')
    #
    #     axs[1].plot(controls_kalman['tau'][dof, :].T, color='blue')
    #     axs[1].plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
    #     axs[1].plot(controls['tau'][dof, :].T, color='green')
    #
    #     fig.suptitle(dofs_name[idx_dof])
    #     lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    #     lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    #     lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    #     axs[0].legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])
    #
    #     axs[0].set_ylabel(r"$\mathregular{rad}$")
    #     axs[0].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    #     axs[0].tick_params(axis="y", direction='in')
    #     axs[0].spines['right'].set_visible(False)
    #     axs[0].spines['top'].set_visible(False)
    #
    #     axs[1].set_xlabel("Aerial time")
    #     axs[1].set_ylabel(r"$\mathregular{N \cdot m}$")
    #     axs[1].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    #     axs[1].tick_params(axis="y", direction='in')
    #     axs[1].spines['right'].set_visible(False)
    #     axs[1].spines['top'].set_visible(False)
    #
    #     save_path = 'Solutions/'
    #     save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_' + dofs_name[idx_dof] + '.png'
    #     pyplot.savefig(save_name)

    pyplot.show()

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
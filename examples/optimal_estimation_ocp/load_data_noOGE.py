import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function, sum1
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
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import correct_Kalman, check_Kalman, shift_by_2pi

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


def states_to_markers(biorbd_model, ocp, states):
    q = states['q']
    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "markers_func", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    return markers


def states_to_markers_velocity(biorbd_model, ocp, states):
    q = states['q']
    qdot = states['q_dot']
    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers_velocity = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_q = MX.sym("q", n_q, 1)
    symbolic_qdot = MX.sym("qdot", n_qdot, 1)
    # This doesn't work for some mysterious reasons
    # markers_func = Function(
    #     "markers_func", [symbolic_q, symbolic_qdot], [biorbd_model.markersVelocity(symbolic_q, symbolic_qdot)], ["q", "q_dot"], ["markers_velocity"]
    # ).expand()
    for j in range(n_mark):
        markers_func = biorbd.to_casadi_func('markers_func', biorbd_model.markerVelocity, symbolic_q, symbolic_qdot, j)
        for i in range(n_frames):
            markers_velocity[:, j, i] = markers_func(q[:, i], qdot[:, i]).full().squeeze()

    return markers_velocity


def rank_jacobian_marker_state(biorbd_model, state, markers_mocap):
    symbolic_states = MX.sym("x", state.shape[0], 1)
    markers_func = Function(
        "markers_func", [symbolic_states], [sum1((biorbd_model.markers(symbolic_states) - markers_mocap)**2)], ["q"], ["markers"]
    ).expand()

    jac = markers_func.jacobian()

    # return np.linalg.matrix_rank(np.nan_to_num(jac(state, 0).full()))
    # return np.linalg.matrix_rank(jac(state, 0).full()[~np.isnan(jac(state, 0).full()).any(axis=1)])

    # --- Explicit np.linalg.matrix_rank, to get missing DoF --- #
    M = np.asarray(jac(state, 0).full()[~np.isnan(jac(state, 0).full()).any(axis=1)])
    S = np.linalg.svd(M, compute_uv=False, hermitian=False)
    tol = S.max(axis=-1, keepdims=True) * np.max(M.shape[-2:]) * np.finfo(S.dtype).eps
    return np.count_nonzero(S > tol, axis=-1), (S.shape, np.where(S <= tol))


def stats(subject, trial, number_shooting_points):
    print('Subject: ', subject)
    print('Trial: ', trial)

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    biorbd_model = biorbd.Model(model_path + model_name)
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Functions --- #
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)
    am = biorbd.to_casadi_func("am", biorbd_model.CalcAngularMomentum, q, qdot, qddot, True)
    fd = biorbd.to_casadi_func("fd", biorbd_model.ForwardDynamics, q, qdot, tau)
    mcm = biorbd.to_casadi_func("fd", biorbd_model.mass)
    vcm = biorbd.to_casadi_func("fd", biorbd_model.CoMdot, q, qdot)


    # --- Load --- #
    load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + "_noOGE"
    ocp, sol = OptimalControlProgram.load(load_name + ".bo")

    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    markers_mocap = data['mocap']
    frames = data['frames']
    step_size = data['step_size']

    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    optimal_gravity_filename = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_noOGE' + ".bo"

    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)
    # test_Kalman = np.copy(kalman_states['q'][:, ::step_size])
    # q_kalman_biorbd = correct_Kalman(biorbd_model, kalman_states['q'][:, ::step_size])
    # q_kalman_biorbd = test_Kalman
    q_kalman_biorbd = shift_by_2pi(biorbd_model, kalman_states['q'][:, ::step_size])
    # q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

    states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
    controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)}

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)
    qddot = fd(states['q'], states['q_dot'], controls['tau'])

    ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    states_optimal_gravity, controls_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity)
    qddot_optimal_gravity = fd(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], controls_optimal_gravity['tau'])

    markers = states_to_markers(biorbd_model, ocp, states)
    markers_optimal_gravity = states_to_markers(biorbd_model, ocp, states_optimal_gravity)
    markers_kalman_biorbd = states_to_markers(biorbd_model, ocp, states_kalman_biorbd)

    markers_velocity_OE = states_to_markers_velocity(biorbd_model, ocp, states)
    markers_velocity_OGE = states_to_markers_velocity(biorbd_model, ocp, states_optimal_gravity)
    markers_velocity_EKF_biorbd = states_to_markers_velocity(biorbd_model, ocp, states_kalman_biorbd)

    # --- Marker error --- #
    markers_error_OE = np.sqrt(np.sum((markers - markers_mocap) ** 2, axis=0)) * 1000
    markers_error_OGE = np.sqrt(np.sum((markers_optimal_gravity - markers_mocap) ** 2, axis=0)) * 1000
    markers_error_EKF_biorbd = np.sqrt(np.sum((markers_kalman_biorbd - markers_mocap) ** 2, axis=0)) * 1000

    # --- Missing markers --- #
    missing_markers_per_node = np.count_nonzero(np.isnan(markers_mocap[0, :, :]), axis=0) / markers_mocap.shape[1] * 100

    # --- By segments --- #
    segments = dict()
    dof_idx = 0
    for segment_idx in range(biorbd_model.nbSegment()):
        segments[biorbd_model.segment(segment_idx).id()] = {
            'name': biorbd_model.segment(segment_idx).name().to_string(),
            'markers_idx': [],
            'dofs': range(dof_idx, dof_idx + biorbd_model.segment(segment_idx).nbDof())}
        dof_idx += biorbd_model.segment(segment_idx).nbDof()
    for marker_idx in range(biorbd_model.nbMarkers()):
        segments[biorbd_model.marker(marker_idx).parentId()]['markers_idx'].append(marker_idx)

    segment_names = []
    segment_marker_error_OE = {}
    segment_marker_error_OGE = {}
    segment_marker_error_EKF_biorbd = {}
    segment_marker_error_OE_all = {}
    segment_marker_error_OGE_all = {}
    segment_marker_error_EKF_biorbd_all = {}
    segment_marker_error_OE_per_node = {}
    segment_marker_error_OGE_per_node = {}
    segment_marker_error_EKF_biorbd_per_node = {}
    segment_missing_markers = {}
    segment_missing_markers_per_node = {}
    for segment in segments.values():
        segment_names.append(segment['name'])
        segment_marker_error_OE[segment['name']] = np.nanmean([markers_error_OE[marker_idx, :] for marker_idx in segment['markers_idx']])
        segment_marker_error_OGE[segment['name']] = np.nanmean([markers_error_OGE[marker_idx, :] for marker_idx in segment['markers_idx']])
        segment_marker_error_EKF_biorbd[segment['name']] = np.nanmean([markers_error_EKF_biorbd[marker_idx, :] for marker_idx in segment['markers_idx']])

        segment_marker_error_OE_all[segment['name']] = [markers_error_OE[marker_idx, :] for marker_idx in segment['markers_idx']]
        segment_marker_error_OGE_all[segment['name']] = [markers_error_OGE[marker_idx, :] for marker_idx in segment['markers_idx']]
        segment_marker_error_EKF_biorbd_all[segment['name']] = [markers_error_EKF_biorbd[marker_idx, :] for marker_idx in segment['markers_idx']]

        segment_marker_error_OE_per_node[segment['name']] = np.nanmean([markers_error_OE[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0)
        segment_marker_error_OGE_per_node[segment['name']] = np.nanmean([markers_error_OGE[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0)
        segment_marker_error_EKF_biorbd_per_node[segment['name']] = np.nanmean([markers_error_EKF_biorbd[marker_idx, :] for marker_idx in segment['markers_idx']], axis=0)

        segment_missing_markers[segment['name']] = np.count_nonzero(np.isnan([markers_mocap[0, marker_idx, :] for marker_idx in segment['markers_idx']])) / len(segment['markers_idx']) / markers_mocap.shape[2] * 100
        segment_missing_markers_per_node[segment['name']] = np.count_nonzero(np.isnan([markers_mocap[0, marker_idx, :] for marker_idx in segment['markers_idx']]), axis=0) / len(segment['markers_idx']) * 100

    trunk_segments = ['Pelvis', 'Thorax', 'Tete']
    arms_segments = ['EpauleD', 'BrasD', 'ABrasD', 'MainD', 'EpauleG', 'BrasG', 'ABrasG', 'MainG']
    legs_segments = ['CuisseD', 'JambeD', 'PiedD', 'CuisseG', 'JambeG', 'PiedG']

    trunk_marker_error_OE = [segment_marker_error_OE_all[segment] for segment in trunk_segments]
    arms_marker_error_OE = [segment_marker_error_OE_all[segment] for segment in arms_segments]
    legs_marker_error_OE = [segment_marker_error_OE_all[segment] for segment in legs_segments]

    trunk_marker_error_OGE = [segment_marker_error_OGE_all[segment] for segment in trunk_segments]
    arms_marker_error_OGE = [segment_marker_error_OGE_all[segment] for segment in arms_segments]
    legs_marker_error_OGE = [segment_marker_error_OGE_all[segment] for segment in legs_segments]

    trunk_marker_error_EKF_biorbd = [segment_marker_error_EKF_biorbd_all[segment] for segment in trunk_segments]
    arms_marker_error_EKF_biorbd = [segment_marker_error_EKF_biorbd_all[segment] for segment in arms_segments]
    legs_marker_error_EKF_biorbd = [segment_marker_error_EKF_biorbd_all[segment] for segment in legs_segments]

    # --- Stats --- #
    flatten = lambda t: [item for sublist in t for item in sublist]

    #OE
    average_distance_between_markers = np.nanmean(markers_error_OE)
    sd_distance_between_markers = np.nanstd(markers_error_OE)

    average_distance_between_markers_OE_trunk = np.nanmean(flatten(trunk_marker_error_OE))
    sd_distance_between_markers_OE_trunk = np.nanstd(flatten(trunk_marker_error_OE))
    average_distance_between_markers_OE_arms = np.nanmean(flatten(arms_marker_error_OE))
    sd_distance_between_markers_OE_arms = np.nanstd(flatten(arms_marker_error_OE))
    average_distance_between_markers_OE_legs = np.nanmean(flatten(legs_marker_error_OE))
    sd_distance_between_markers_OE_legs = np.nanstd(flatten(legs_marker_error_OE))

    # OGE
    average_distance_between_markers_optimal_gravity = np.nanmean(markers_error_OGE)
    sd_distance_between_markers_optimal_gravity = np.nanstd(markers_error_OGE)

    average_distance_between_markers_OGE_trunk = np.nanmean(flatten(trunk_marker_error_OGE))
    sd_distance_between_markers_OGE_trunk = np.nanstd(flatten(trunk_marker_error_OGE))
    average_distance_between_markers_OGE_arms = np.nanmean(flatten(arms_marker_error_OGE))
    sd_distance_between_markers_OGE_arms = np.nanstd(flatten(arms_marker_error_OGE))
    average_distance_between_markers_OGE_legs = np.nanmean(flatten(legs_marker_error_OGE))
    sd_distance_between_markers_OGE_legs = np.nanstd(flatten(legs_marker_error_OGE))

    # EKF biorbd
    average_distance_between_markers_kalman_biorbd = np.nanmean(markers_error_EKF_biorbd)
    sd_distance_between_markers_kalman_biorbd = np.nanstd(markers_error_EKF_biorbd)

    average_distance_between_markers_EKF_biorbd_trunk = np.nanmean(flatten(trunk_marker_error_EKF_biorbd))
    sd_distance_between_markers_EKF_biorbd_trunk = np.nanstd(flatten(trunk_marker_error_EKF_biorbd))
    average_distance_between_markers_EKF_biorbd_arms = np.nanmean(flatten(arms_marker_error_EKF_biorbd))
    sd_distance_between_markers_EKF_biorbd_arms = np.nanstd(flatten(arms_marker_error_EKF_biorbd))
    average_distance_between_markers_EKF_biorbd_legs = np.nanmean(flatten(legs_marker_error_EKF_biorbd))
    sd_distance_between_markers_EKF_biorbd_legs = np.nanstd(flatten(legs_marker_error_EKF_biorbd))

    RMSE_difference_between_Q_OGE = np.sqrt(np.mean((states_kalman_biorbd['q'] - states_optimal_gravity['q']) ** 2))
    RMSE_difference_between_Q_OE = np.sqrt(np.mean((states_kalman_biorbd['q'] - states['q']) ** 2))

    trunk_dofs = range(0, 12)
    arms_dofs = range(12, 30)
    legs_dofs = range(30, 42)

    RMSE_difference_between_Q_OGE_trunk = np.sqrt(np.mean((states_kalman_biorbd['q'] - states_optimal_gravity['q'])[trunk_dofs, :] ** 2))
    RMSE_difference_between_Q_OE_trunk = np.sqrt(np.mean((states_kalman_biorbd['q'] - states['q'])[trunk_dofs, :] ** 2))
    RMSE_difference_between_Q_OGE_arms = np.sqrt(np.mean((states_kalman_biorbd['q'] - states_optimal_gravity['q'])[arms_dofs, :] ** 2))
    RMSE_difference_between_Q_OE_arms = np.sqrt(np.mean((states_kalman_biorbd['q'] - states['q'])[arms_dofs, :] ** 2))
    RMSE_difference_between_Q_OGE_legs = np.sqrt(np.mean((states_kalman_biorbd['q'] - states_optimal_gravity['q'])[legs_dofs, :] ** 2))
    RMSE_difference_between_Q_OE_legs = np.sqrt(np.mean((states_kalman_biorbd['q'] - states['q'])[legs_dofs, :] ** 2))

    RMSE_difference_between_Q_OE_segment = []
    RMSE_difference_between_Q_OGE_segment = []
    for segment in segments.values():
        RMSE_difference_between_Q_OE_segment.append(np.sqrt(np.mean((states_kalman_biorbd['q'][segment['dofs']] - states['q'][segment['dofs']]) ** 2)))
        RMSE_difference_between_Q_OGE_segment.append(np.sqrt(np.mean((states_kalman_biorbd['q'][segment['dofs']] - states_optimal_gravity['q'][segment['dofs']]) ** 2)))


    momentum_OE = am(states['q'], states['q_dot'], qddot).full()
    momentum_OGE = am(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], qddot_optimal_gravity).full()
    momentum_EKF = am(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()
    # momentum_sim_OGE = am(states_OGE_sim['q'], states_OGE_sim['q_dot'], qddot_OGE_sim).full()
    # momentum_sim_OE = am(states_OE_sim['q'], states_OE_sim['q_dot'], qddot_OE_sim).full()

    # median_OE = np.median(momentum_OE, axis=1)
    # median_OGE = np.median(momentum_OGE, axis=1)
    # median_EKF = np.median(momentum_EKF, axis=1)
    #
    # mad_OE = np.median(np.abs(momentum_OE - median_OE[:, np.newaxis]), axis=1)
    # mad_OGE = np.median(np.abs(momentum_OGE - median_OGE[:, np.newaxis]), axis=1)
    # mad_EKF = np.median(np.abs(momentum_EKF - median_EKF[:, np.newaxis]), axis=1)

    mean_OE = np.mean(momentum_OE, axis=1)[:, np.newaxis]
    mean_OGE = np.mean(momentum_OGE, axis=1)[:, np.newaxis]
    mean_EKF = np.mean(momentum_EKF, axis=1)[:, np.newaxis]

    RMSE_momentum_OE = np.sqrt(np.mean((momentum_OE - mean_OE) ** 2, axis=1))
    RMSE_momentum_OGE = np.sqrt(np.mean((momentum_OGE - mean_OGE) ** 2, axis=1))
    RMSE_momentum_EKF = np.sqrt(np.mean((momentum_EKF - mean_EKF) ** 2, axis=1))

    total_mass = mcm()['o0'].full()
    linear_momentum = total_mass * vcm(states['q'], states['q_dot']).full()
    linear_momentum_optimal_gravity = total_mass * vcm(states_optimal_gravity['q'], states_optimal_gravity['q_dot']).full()
    linear_momentum_kalman = total_mass * vcm(q_kalman_biorbd, qdot_kalman_biorbd).full()
    # linear_momentum_sim_OGE = total_mass * vcm(states_OGE_sim['q'], states_OGE_sim['q_dot']).full()
    # linear_momentum_sim_OE = total_mass * vcm(states_OE_sim['q'], states_OE_sim['q_dot']).full()

    slope_lm, intercept_lm = np.polyfit(range(linear_momentum.shape[1]), linear_momentum.T, 1)
    slope_lm_optimal_gravity, intercept_lm_optimal_gravity = np.polyfit(range(linear_momentum_optimal_gravity.shape[1]), linear_momentum_optimal_gravity.T, 1)
    slope_lm_kalman, intercept_lm_kalman = np.polyfit(range(linear_momentum_kalman.shape[1]), linear_momentum_kalman.T, 1)

    # diff_lm = (linear_momentum[:, 1:] - linear_momentum[:, :-1])/total_mass/(duration/adjusted_number_shooting_points)
    # diff_lm_optimal_gravity = (linear_momentum_optimal_gravity[:, 1:] - linear_momentum_optimal_gravity[:, :-1])/total_mass/(duration/adjusted_number_shooting_points)
    # diff_lm_kalman = (linear_momentum_kalman[:, 1:] - linear_momentum_kalman[:, :-1])/total_mass/(duration/adjusted_number_shooting_points)

    predicted_lm_OE = np.outer(slope_lm, range(linear_momentum.shape[1])) + intercept_lm[:, np.newaxis]
    predicted_lm_OGE = np.outer(slope_lm_optimal_gravity, range(linear_momentum.shape[1])) + intercept_lm_optimal_gravity[:, np.newaxis]
    predicted_lm_EKF = np.outer(slope_lm_kalman, range(linear_momentum.shape[1])) + intercept_lm_kalman[:, np.newaxis]

    RMSE_linear_momentum_OE = np.sqrt(np.mean((linear_momentum - predicted_lm_OE) ** 2, axis=1))
    RMSE_linear_momentum_OGE = np.sqrt(np.mean((linear_momentum_optimal_gravity - predicted_lm_OGE) ** 2, axis=1))
    RMSE_linear_momentum_EKF = np.sqrt(np.mean((linear_momentum_kalman - predicted_lm_EKF) ** 2, axis=1))

    rank = []
    for i in range(adjusted_number_shooting_points):
        node_rank, node_incomplete_dof = rank_jacobian_marker_state(biorbd_model, states['q'][:, i], markers_mocap[:, :, i])
        rank.append(node_rank)
        # if rank[i] != biorbd_model.nbQ():
        #     print(i, rank[i],  node_incomplete_dof)
    # print(rank)
    # print(rank.count(biorbd_model.nbQ()) / adjusted_number_shooting_points * 100)


    # --- Markers velocity --- #

    model_labels = [label.to_string() for label in biorbd_model.markerNames()]

    norm_markers_velocity_OE = np.linalg.norm(markers_velocity_OE, axis=0)
    norm_markers_velocity_OGE = np.linalg.norm(markers_velocity_OGE, axis=0)
    norm_markers_velocity_EKF_biorbd = np.linalg.norm(markers_velocity_EKF_biorbd, axis=0)

    idx_max_all_velocity_OE = np.argmax(norm_markers_velocity_OE, axis=0)
    idx_max_velocity_OE = np.unravel_index(norm_markers_velocity_OE.argmax(), norm_markers_velocity_OE.shape)
    idx_max_middle_velocity_OE = np.unravel_index(
        norm_markers_velocity_OE[:, int(adjusted_number_shooting_points/4):int(adjusted_number_shooting_points*3/4)].argmax(),
        norm_markers_velocity_OE[:, int(adjusted_number_shooting_points/4):int(adjusted_number_shooting_points*3/4)].shape)
    idx_min_all_velocity_OE = np.argmin(norm_markers_velocity_OE, axis=0)
    idx_min_velocity_OE = np.unravel_index(norm_markers_velocity_OE.argmin(), norm_markers_velocity_OE.shape)
    idx_min_middle_velocity_OE = np.unravel_index(
        norm_markers_velocity_OE[:, int(adjusted_number_shooting_points/4):int(adjusted_number_shooting_points*3/4)].argmin(),
        norm_markers_velocity_OE[:, int(adjusted_number_shooting_points/4):int(adjusted_number_shooting_points*3/4)].shape)

    # --- Plots --- #

    fig_momentum, axs_momentum = pyplot.subplots(nrows=2, ncols=1, figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1]})
    axs_momentum[0].plot(momentum_OE.T, color='green')
    axs_momentum[0].plot(momentum_OGE.T, color='red', linestyle=':')
    axs_momentum[0].plot(momentum_EKF.T, color='blue')
    # axs_momentum[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), momentum_sim_OGE.T, color='grey', linestyle=':')
    # axs_momentum[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), momentum_sim_OE.T, color='black', linestyle='--')

    lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='red')
    lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    fig_momentum.legend([lm_oe, lm_og, lm_kal], ['Kalman', 'OGE', 'OE'])
    # fig_momentum.title('Angular momentum of free fall movement')
    # fig_momentum.xlabel("Aerial time")
    axs_momentum[0].set_ylabel(r"$\mathregular{kg \cdot m^2/s}$")
    axs_momentum[0].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    axs_momentum[0].tick_params(axis="y", direction='in')
    axs_momentum[0].spines['right'].set_visible(False)
    axs_momentum[0].spines['top'].set_visible(False)

    axs_momentum[0].annotate('x', (momentum_OE.shape[1], momentum_OE.T[-1, 0]), textcoords="offset points", xytext=(2, 0), ha='center')
    axs_momentum[0].annotate('y', (momentum_OE.shape[1], momentum_OE.T[-1, 1]), textcoords="offset points", xytext=(2, 0), ha='center')
    axs_momentum[0].annotate('z', (momentum_OE.shape[1], momentum_OE.T[-1, 2]), textcoords="offset points", xytext=(2, 0), ha='center')

    box_text = (
            'EKF RMSE: x: ' + np.format_float_scientific(RMSE_momentum_EKF[0], precision=1, exp_digits=2) +
                   ', y: ' + np.format_float_scientific(RMSE_momentum_EKF[1], precision=1, exp_digits=2) +
                   ', z: ' + np.format_float_scientific(RMSE_momentum_EKF[2], precision=1, exp_digits=2) + '\n' +
            'OGE RMSE: x: ' + np.format_float_scientific(RMSE_momentum_OGE[0], precision=1, exp_digits=2) +
                   ', y: ' + np.format_float_scientific(RMSE_momentum_OGE[1], precision=1, exp_digits=2) +
                   ', z: ' + np.format_float_scientific(RMSE_momentum_OGE[2], precision=1, exp_digits=2) + '\n' +
            'OE RMSE: x: ' + np.format_float_scientific(RMSE_momentum_OE[0], precision=1, exp_digits=2) +
                  ', y: ' + np.format_float_scientific(RMSE_momentum_OE[1], precision=1, exp_digits=2) +
                  ', z: ' + np.format_float_scientific(RMSE_momentum_OE[2], precision=1, exp_digits=2)
    )
    text_box = AnchoredText(box_text, frameon=True, loc=5, pad=0.5, prop=dict(fontsize=9))
    pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
    axs_momentum[0].add_artist(text_box)

    axs_momentum[1].plot(linear_momentum.T, color='green')
    axs_momentum[1].plot(linear_momentum_optimal_gravity.T, color='red', linestyle=':')
    axs_momentum[1].plot(linear_momentum_kalman.T, color='blue')
    # axs_momentum[1].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), linear_momentum_sim_OGE.T, color='grey', linestyle=':')
    # axs_momentum[1].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), linear_momentum_sim_OE.T, color='black', linestyle='--')

    # lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    # lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='red')
    # lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    # fig_momentum.legend([lm_kal, lm_og, lm_oe], ['EKF', 'OGE', 'OE'])
    # fig_momentum.title('Linear momentum of free fall movement')
    axs_momentum[1].set_ylabel(r"$\mathregular{kg \cdot m/s}$")
    axs_momentum[1].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    axs_momentum[1].tick_params(axis="y", direction='in')
    axs_momentum[1].spines['right'].set_visible(False)
    axs_momentum[1].spines['top'].set_visible(False)

    axs_momentum[1].annotate('x', (linear_momentum.shape[1], linear_momentum.T[-1, 0]), textcoords="offset points", xytext=(2, -2), ha='center')
    axs_momentum[1].annotate('y', (linear_momentum.shape[1], linear_momentum.T[-1, 1]), textcoords="offset points", xytext=(2, 2), ha='center')
    axs_momentum[1].annotate('z', (linear_momentum.shape[1], linear_momentum.T[-1, 2]), textcoords="offset points", xytext=(2, 0), ha='center')

    box_text = (
            'EKF $\mathregular{G_{norm}}$: ' + f"{np.linalg.norm(slope_lm_kalman / total_mass / (duration / adjusted_number_shooting_points)):.4f}" + '\t' +
            'EKF RMSE: x: ' + np.format_float_scientific(RMSE_linear_momentum_EKF[0], precision=1, exp_digits=2) +
                   ', y: ' + np.format_float_scientific(RMSE_linear_momentum_EKF[1], precision=1, exp_digits=2) +
                   ', z: ' + np.format_float_scientific(RMSE_linear_momentum_EKF[2], precision=1, exp_digits=2) + '\n' +
            'OGE $\mathregular{G_{norm}}$: ' + f"{np.linalg.norm(slope_lm_optimal_gravity/total_mass/(duration/adjusted_number_shooting_points)):.4f}" + '\t' +
            'OGE RMSE: x: ' + np.format_float_scientific(RMSE_linear_momentum_OGE[0], precision=1, exp_digits=2) +
                   ', y: ' + np.format_float_scientific(RMSE_linear_momentum_OGE[1], precision=1, exp_digits=2) +
                   ', z: ' + np.format_float_scientific(RMSE_linear_momentum_OGE[2], precision=1, exp_digits=2) + '\n' +
            'OE $\mathregular{G_{norm}}$: ' + f"{np.linalg.norm(slope_lm/total_mass/(duration/adjusted_number_shooting_points)):.4f}" + '\t' +
            'OE RMSE: x: ' + np.format_float_scientific(RMSE_linear_momentum_OE[0], precision=1, exp_digits=2) +
                  ', y: ' + np.format_float_scientific(RMSE_linear_momentum_OE[1], precision=1, exp_digits=2) +
                  ', z: ' + np.format_float_scientific(RMSE_linear_momentum_OE[2], precision=1, exp_digits=2)
    )
    text_box = AnchoredText(box_text, frameon=True, loc=1, pad=0.5, prop=dict(fontsize=9))
    pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
    axs_momentum[1].add_artist(text_box)

    # --- Missing markers --- #
    fig_error, axs_error = pyplot.subplots(nrows=2, ncols=1, figsize=(20, 10), gridspec_kw={'height_ratios': [1/4, 1]})

    nb_nan = np.count_nonzero(np.isnan(markers_mocap[0, :, :]), axis=0) / markers_mocap.shape[1] * 100 # Percentage
    im = axs_error[0].imshow(nb_nan[np.newaxis, :], cmap="plasma", aspect="auto")

    # axs_error[0].set_ylabel('% of missing markers', rotation=0, fontsize=9, horizontalalignment='right')
    axs_error[0].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    axs_error[0].tick_params(axis="y", direction='in', which='both', left=False, right=False, labelleft=False)

    axs_error[0].spines['right'].set_visible(False)
    axs_error[0].spines['left'].set_visible(False)
    axs_error[0].spines['top'].set_visible(False)
    axs_error[0].spines['bottom'].set_visible(False)

    cbar = fig_error.colorbar(im, ax=axs_error.ravel().tolist(), shrink=0.95, pad=0.1)
    cbar.ax.set_ylabel('% of missing markers', rotation=270, labelpad=15)
    cbar.ax.tick_params(axis="y", direction='in', which='both')

    # --- Marker error --- #
    axs_error[1].plot(np.nanmean(markers_error_OE, axis=0), color='green')
    axs_error[1].plot(np.nanmean(markers_error_OGE, axis=0), linestyle=':', color='red')
    axs_error[1].plot(np.nanmean(markers_error_EKF_biorbd, axis=0), color='blue')

    axs_error[1].set_ylabel("Marker error" "\n" r"$\mathregular{mm}$")
    axs_error[1].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    axs_error[1].tick_params(axis="y", direction='in')
    axs_error[1].spines['right'].set_visible(False)
    axs_error[1].spines['top'].set_visible(False)

    axs_error[0].set_xlim(axs_error[1].get_xlim())

    axs_error[-1].set_xlabel("Aerial time")

    pyplot.margins(0, 0)
    fig_momentum.gca().xaxis.set_major_locator(pyplot.NullLocator())
    fig_momentum.gca().yaxis.set_major_locator(pyplot.NullLocator())
    save_path = 'Solutions/'
    save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + "_momentum_noOGE" + '.png'
    fig_momentum.savefig(save_name, bbox_inches='tight', pad_inches=0)

    pyplot.margins(0, 0)
    fig_error.gca().xaxis.set_major_locator(pyplot.NullLocator())
    fig_error.gca().yaxis.set_major_locator(pyplot.NullLocator())
    save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + "_error_noOGE" + '.png'
    fig_error.savefig(save_name, bbox_inches='tight', pad_inches=0)

    fig_marker_velocity = pyplot.figure(figsize=(20, 10))
    pyplot.plot(norm_markers_velocity_OE.T, color='blue')
    fig_marker_velocity = pyplot.figure(figsize=(20, 10))
    pyplot.plot(norm_markers_velocity_OGE.T, color='red')
    fig_marker_velocity = pyplot.figure(figsize=(20, 10))
    pyplot.plot(norm_markers_velocity_EKF_biorbd.T, color='green')

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
    # dofs = range(0, 6)
    fig_model_dof = [(4, 2), (1, 2), (0, 2),
                     (1, 0), (2, 0), (3, 0), (4, 0),
                     (1, 4), (2, 4), (3, 4), (4, 4),
                     (5, 1), (6, 1), (7, 1),
                     (5, 3), (6, 3), (7, 3)]
    # fig_model, axs_model = pyplot.subplots(nrows=8, ncols=6, figsize=(20, 10))
    fig_model_Q = pyplot.figure(figsize=(20, 10))
    fig_model_U = pyplot.figure(figsize=(20, 10))
    fig_model_error_missing = pyplot.figure(figsize=(20, 10))
    gs_model = gridspec.GridSpec(8, 6)
    for idx_dof, dof in enumerate(dofs):
        # fig, axs = pyplot.subplots(nrows=4, ncols=1, figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1, 1/4, 1]})
        # axs[0].plot(states_kalman_biorbd['q'][dof, :].T, color='blue')
        # # axs[0].plot(test_Kalman[dof, :].T, color='blue', linestyle=':')
        # axs[0].plot(states_optimal_gravity['q'][dof, :].T, color='red')
        # axs[0].plot(states['q'][dof, :].T, color='green')
        # # axs[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), states_OGE_sim['q'][dof, :].T, color='grey', linestyle=':')
        # # axs[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), states_OE_sim['q'][dof, :].T, color='black', linestyle='--')
        #
        # axs[1].plot(controls_kalman_biorbd['tau'][dof, :].T, color='blue')
        # axs[1].plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
        # axs[1].plot(controls['tau'][dof, :].T, color='green')
        #
        # fig_qdot = pyplot.figure()
        # pyplot.plot(states_kalman_biorbd['q_dot'][dof, :].T, color='blue')
        # pyplot.plot(states_optimal_gravity['q_dot'][dof, :].T, color='red')
        # pyplot.plot(states['q_dot'][dof, :].T, color='green')
        # fig_qdot.suptitle(dofs_name[idx_dof])
        # #
        # # fig = pyplot.figure()
        # # pyplot.plot(qddot_ref_biorbd[dof, :].T, color='red')
        # #
        #
        # fig.suptitle(dofs_name[idx_dof])
        # lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        # lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
        # lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        # fig.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])
        #
        # axs[0].set_ylabel(r"$\mathregular{rad}$")
        # axs[0].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        # axs[0].tick_params(axis="y", direction='in')
        # axs[0].spines['right'].set_visible(False)
        # axs[0].spines['top'].set_visible(False)
        #
        # axs[1].set_ylabel(r"$\mathregular{N \cdot m}$")
        # axs[1].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        # axs[1].tick_params(axis="y", direction='in')
        # axs[1].spines['right'].set_visible(False)
        # axs[1].spines['top'].set_visible(False)
        # axs[1].set_ylim([-300, 300])
        #
        # # --- Heatmap of missing markers --- #
        # im = axs[2].imshow(segment_missing_markers_per_node[segment_names[idx_dof]][np.newaxis, :], cmap="plasma", aspect="auto")
        #
        # # axs[2].set_ylabel('% of missing markers', rotation=0, fontsize=9, horizontalalignment='right')
        # axs[2].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        # axs[2].tick_params(axis="y", direction='in', which='both', left=False, right=False, labelleft=False)
        #
        # axs[2].set_xlim(axs[1].get_xlim())
        # axs[2].spines['right'].set_visible(False)
        # axs[2].spines['left'].set_visible(False)
        # axs[2].spines['top'].set_visible(False)
        # axs[2].spines['bottom'].set_visible(False)
        #
        # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95, pad=0.1)
        # cbar.ax.set_ylabel('% of missing markers', rotation=270, labelpad=15)
        # cbar.ax.tick_params(axis="y", direction='in', which='both')
        #
        # # --- Marker error --- #
        # axs[3].plot(segment_marker_error_EKF_biorbd_per_node[segment_names[idx_dof]], color='blue')
        # axs[3].plot(segment_marker_error_OGE_per_node[segment_names[idx_dof]], color='red')
        # axs[3].plot(segment_marker_error_OE_per_node[segment_names[idx_dof]], color='green')
        #
        # axs[3].set_ylabel(r"$\mathregular{mm}$")
        # axs[3].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        # axs[3].tick_params(axis="y", direction='in')
        # axs[3].spines['right'].set_visible(False)
        # axs[3].spines['top'].set_visible(False)
        #
        # axs[-1].set_xlabel("Aerial time")

        # --- Model subplots --- #
        gs_model_subplot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1]+2])

        ax_model_box = fig_model_Q.add_subplot(gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
        ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        ax_model_box.patch.set_alpha(0.3)

        ax_model_Q = fig_model_Q.add_subplot(gs_model_subplot[0])
        ax_model_Q.plot(states_kalman_biorbd['q'][dof, :].T, color='blue')
        ax_model_Q.plot(states_optimal_gravity['q'][dof, :].T, color='red')
        ax_model_Q.plot(states['q'][dof, :].T, color='green')

        ax_model_Q.set_title(dofs_name[idx_dof], size=9)
        ax_model_Q.set_ylabel(r"$\mathregular{rad}$", size=9)
        ax_model_Q.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        ax_model_Q.tick_params(axis="y", direction='in', labelsize=9)
        ax_model_Q.spines['right'].set_visible(False)
        ax_model_Q.spines['top'].set_visible(False)
        ax_model_Q.spines['bottom'].set_visible(False)

        ax_model_box = fig_model_U.add_subplot(gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
        ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        ax_model_box.patch.set_alpha(0.3)

        ax_model_U = fig_model_U.add_subplot(gs_model_subplot[0])
        ax_model_U.plot(controls_kalman_biorbd['tau'][dof, :].T, color='blue')
        ax_model_U.plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
        ax_model_U.plot(controls['tau'][dof, :].T, color='green')

        ax_model_U.set_title(dofs_name[idx_dof], size=9)
        ax_model_U.set_ylabel(r"$\mathregular{rad}$", size=9)
        ax_model_U.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        ax_model_U.tick_params(axis="y", direction='in', labelsize=9)
        ax_model_U.spines['right'].set_visible(False)
        ax_model_U.spines['top'].set_visible(False)
        ax_model_U.spines['bottom'].set_visible(False)


        gs_model_subplot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1]+2], height_ratios=[1/6, 1])

        ax_model_box = fig_model_error_missing.add_subplot(gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
        ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        ax_model_box.patch.set_alpha(0.3)

        ax_model_1 = fig_model_error_missing.add_subplot(gs_model_subplot[0])
        im_model = ax_model_1.imshow(segment_missing_markers_per_node[segment_names[idx_dof]][np.newaxis, :], cmap="plasma", aspect="auto", vmin=0, vmax=50)
        ax_model_1.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        ax_model_1.tick_params(axis="y", direction='in', which='both', left=False, right=False, labelleft=False)

        ax_model_1.set_title(dofs_name[idx_dof], size=9)
        ax_model_1.spines['right'].set_visible(False)
        ax_model_1.spines['left'].set_visible(False)
        ax_model_1.spines['top'].set_visible(False)
        ax_model_1.spines['bottom'].set_visible(False)

        ax_model_2 = fig_model_error_missing.add_subplot(gs_model_subplot[1])
        ax_model_2.plot(segment_marker_error_EKF_biorbd_per_node[segment_names[idx_dof]], color='blue')
        ax_model_2.plot(segment_marker_error_OGE_per_node[segment_names[idx_dof]], color='red')
        ax_model_2.plot(segment_marker_error_OE_per_node[segment_names[idx_dof]], color='green')

        ax_model_2.set_ylabel(r"$\mathregular{mm}$", size=9)
        ax_model_2.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        ax_model_2.tick_params(axis="y", direction='in', labelsize=9)
        ax_model_2.spines['right'].set_visible(False)
        ax_model_2.spines['top'].set_visible(False)
        ax_model_2.spines['bottom'].set_visible(False)
        ax_model_1.set_xlim(ax_model_2.get_xlim())

        # ax_model = pyplot.subplot2grid((8, 6), fig_model_dof[idx_dof], colspan=2, fig=fig_model)
        # ax_model.plot(segment_marker_error_OE_per_node[segment_names[idx_dof]], color='blue')
        # ax_model.plot(segment_marker_error_OGE_per_node[segment_names[idx_dof]], color='red')
        # ax_model.plot(segment_marker_error_EKF_biorbd_per_node[segment_names[idx_dof]], color='green')
        #
        # ax_model.title.set_text(dofs_name[idx_dof])
        # ax_model.set_ylabel(r"$\mathregular{mm}$")
        # ax_model.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        # ax_model.tick_params(axis="y", direction='in')
    #     save_path = 'Solutions/'
    #     save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_' + dofs_name[idx_dof] + '.png'
    #     # pyplot.savefig(save_name)

    # for ax in fig_model.get_axes():
    #     if not ax.lines and not ax.images:
    #         fig_model.delaxes(ax)

    # fig_model.suptitle('Subject: ' + subject + ', Trial: ' + trial)

    lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    fig_model_Q.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])
    fig_model_U.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])

    fig_model_Q.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_model_U.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_model_error_missing.subplots_adjust(wspace=0.3, hspace=0.3)

    cbar = fig_model_error_missing.colorbar(im_model, ax=fig_model_error_missing.get_axes(), shrink=0.95, pad=0.1)
    cbar.ax.set_ylabel('% of missing markers', rotation=270, labelpad=15)
    cbar.ax.tick_params(axis="y", direction='in', which='both')

    save_path = 'Solutions/'
    fig_model_Q.tight_layout
    save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_model_Q_noOGE' + '.png'
    # fig_model_Q.savefig(save_name)

    fig_model_U.tight_layout
    save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_model_U_noOGE' + '.png'
    # fig_model_U.savefig(save_name)

    fig_model_error_missing.tight_layout
    save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_model_Error_noOGE' + '.png'
    # fig_model_error_missing.savefig(save_name)

    print('Number of shooting points: ', adjusted_number_shooting_points)
    print('Average marker error')
    print('Kalman biorbd: ', average_distance_between_markers_kalman_biorbd, u"\u00B1", sd_distance_between_markers_kalman_biorbd)
    print('Optimal gravity: ', average_distance_between_markers_optimal_gravity, u"\u00B1", sd_distance_between_markers_optimal_gravity)
    print('Estimation: ', average_distance_between_markers, u"\u00B1", sd_distance_between_markers)

    # print('Max velocity for whole movement: ', model_labels[idx_max_velocity_OE[0]], norm_markers_velocity_OE[idx_max_velocity_OE])
    # print('Max velocity for middle of movement: ', model_labels[idx_max_middle_velocity_OE[0]], norm_markers_velocity_OE[:, int(adjusted_number_shooting_points/4):int(adjusted_number_shooting_points*3/4)][idx_max_middle_velocity_OE])
    # print('Min velocity for whole movement: ', model_labels[idx_min_velocity_OE[0]], norm_markers_velocity_OE[idx_min_velocity_OE])
    # print('Min velocity for middle of movement: ', model_labels[idx_min_middle_velocity_OE[0]], norm_markers_velocity_OE[:, int(adjusted_number_shooting_points / 4):int(adjusted_number_shooting_points * 3 / 4)][idx_min_middle_velocity_OE])
    # print('Mean max velocity: ', np.mean(norm_markers_velocity_OE[idx_max_all_velocity_OE, range(0, idx_max_all_velocity_OE.size)]))
    # print('Mean min velocity: ', np.mean(norm_markers_velocity_OE[idx_min_all_velocity_OE, range(0, idx_min_all_velocity_OE.size)]))
    # print('Mean velocity: ', np.mean(norm_markers_velocity_OE))
    #
    # print('Average percentage of missing markers: ', np.mean(nb_nan))
    #
    #
    # print('EKF linear momentum slope: ', (slope_lm_kalman[0]/total_mass/(duration/adjusted_number_shooting_points))[0][0],
    #                                    (slope_lm_kalman[1]/total_mass/(duration/adjusted_number_shooting_points))[0][0],
    #                                    (slope_lm_kalman[2]/total_mass/(duration/adjusted_number_shooting_points))[0][0],
    #                                    np.linalg.norm(slope_lm_kalman/total_mass/(duration/adjusted_number_shooting_points)))

    # pyplot.show()

    save_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
    save_variables_name = save_path + '.pkl'
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'subject': subject, 'trial': trial, 'adjusted_number_shooting_points': adjusted_number_shooting_points,
                     'frames': frames, 'duration': duration, 'frequency': frequency, 'step_size': step_size,
                     'model_labels': model_labels,
                     'states_EKF': states_kalman_biorbd, 'controls_EKF': controls_kalman_biorbd,
                     'q_EKF': q_kalman_biorbd, 'qdot_EKF': qdot_kalman_biorbd, 'qddot_EKF': qddot_kalman_biorbd,
                     'states_OGE': states_optimal_gravity, 'controls_OGE': controls_optimal_gravity,
                     'q_OGE': states_optimal_gravity['q'], 'qdot_OGE': states_optimal_gravity['q_dot'], 'qddot_OGE': qddot_optimal_gravity.full(),
                     'states_OE': states, 'controls_OE': controls,
                     'q_OE': states['q'], 'qdot_OE': states['q_dot'], 'qddot_OE': qddot.full(),
                     'diff_Q_EKF_OGE': RMSE_difference_between_Q_OGE, 'diff_Q_EKF_OE': RMSE_difference_between_Q_OE,
                     'diff_Q_EKF_OGE_trunk': RMSE_difference_between_Q_OGE_trunk, 'diff_Q_EKF_OE_trunk': RMSE_difference_between_Q_OE_trunk,
                     'diff_Q_EKF_OGE_arms': RMSE_difference_between_Q_OGE_arms, 'diff_Q_EKF_OE_arms': RMSE_difference_between_Q_OE_arms,
                     'diff_Q_EKF_OGE_legs': RMSE_difference_between_Q_OGE_legs, 'diff_Q_EKF_OE_legs': RMSE_difference_between_Q_OE_legs,
                     'markers_mocap': markers_mocap,
                     'markers_EKF': markers_kalman_biorbd,
                     'markers_OGE': markers_optimal_gravity,
                     'markers_OE': markers,
                     'markers_error_EKF': markers_error_EKF_biorbd,
                     'markers_error_OGE': markers_error_OGE,
                     'markers_error_OE': markers_error_OE,
                     'segments': segments, 'segment_names': segment_names,
                     'segment_marker_error_EKF_biorbd': segment_marker_error_EKF_biorbd,
                     'segment_marker_error_OGE': segment_marker_error_OGE,
                     'segment_marker_error_OE': segment_marker_error_OE,
                     'segment_marker_error_EKF_biorbd_per_node': segment_marker_error_EKF_biorbd_per_node,
                     'segment_marker_error_OGE_per_node': segment_marker_error_OGE_per_node,
                     'segment_marker_error_OE_per_node': segment_marker_error_OE_per_node,
                     'segment_missing_markers': segment_missing_markers,
                     'segment_missing_markers_per_node': segment_missing_markers_per_node,
                     'markers_velocity_EKF': markers_velocity_EKF_biorbd,
                     'markers_velocity_OGE': markers_velocity_OGE,
                     'markers_velocity_OE': markers_velocity_OE,
                     'norm_markers_velocity_OE': norm_markers_velocity_OE,
                     'idx_max_velocity_OE': idx_max_velocity_OE, 'idx_max_middle_velocity_OE': idx_max_middle_velocity_OE,
                     'idx_min_velocity_OE': idx_min_velocity_OE, 'idx_min_middle_velocity_OE': idx_min_middle_velocity_OE,
                     'mean_max_velocity': np.mean(norm_markers_velocity_OE[idx_max_all_velocity_OE, range(0, idx_max_all_velocity_OE.size)]),
                     'mean_min_velocity': np.mean(norm_markers_velocity_OE[idx_min_all_velocity_OE, range(0, idx_min_all_velocity_OE.size)]),
                     'mean_veloity': np.mean(norm_markers_velocity_OE),
                     'mean_%_missing_markers': np.mean(nb_nan), 'missing_markers_per_node': missing_markers_per_node,
                     'average_distance_between_markers_EKF_biorbd': average_distance_between_markers_kalman_biorbd,
                     'sd_distance_between_markers_EKF_biorbd': sd_distance_between_markers_kalman_biorbd,
                     'average_distance_between_markers_OGE': average_distance_between_markers_optimal_gravity,
                     'sd_distance_between_markers_OGE': sd_distance_between_markers_optimal_gravity,
                     'average_distance_between_markers_OE': average_distance_between_markers,
                     'sd_distance_between_markers_OE': sd_distance_between_markers,
                     'average_distance_between_markers_EKF_biorbd_trunk': average_distance_between_markers_EKF_biorbd_trunk,
                     'sd_distance_between_markers_EKF_biorbd_trunk': sd_distance_between_markers_EKF_biorbd_trunk,
                     'average_distance_between_markers_EKF_biorbd_arms': average_distance_between_markers_EKF_biorbd_arms,
                     'sd_distance_between_markers_EKF_biorbd_arms': sd_distance_between_markers_EKF_biorbd_arms,
                     'average_distance_between_markers_EKF_biorbd_legs': average_distance_between_markers_EKF_biorbd_legs,
                     'sd_distance_between_markers_EKF_biorbd_legs': sd_distance_between_markers_EKF_biorbd_legs,
                     'average_distance_between_markers_OGE_trunk': average_distance_between_markers_OGE_trunk,
                     'sd_distance_between_markers_OGE_trunk': sd_distance_between_markers_OGE_trunk,
                     'average_distance_between_markers_OGE_arms': average_distance_between_markers_OGE_arms,
                     'sd_distance_between_markers_OGE_arms': sd_distance_between_markers_OGE_arms,
                     'average_distance_between_markers_OGE_legs': average_distance_between_markers_OGE_legs,
                     'sd_distance_between_markers_OGE_legs': sd_distance_between_markers_OGE_legs,
                     'average_distance_between_markers_OE_trunk': average_distance_between_markers_OE_trunk,
                     'sd_distance_between_markers_OE_trunk': sd_distance_between_markers_OE_trunk,
                     'average_distance_between_markers_OE_arms': average_distance_between_markers_OE_arms,
                     'sd_distance_between_markers_OE_arms': sd_distance_between_markers_OE_arms,
                     'average_distance_between_markers_OE_legs': average_distance_between_markers_OE_legs,
                     'sd_distance_between_markers_OE_legs': sd_distance_between_markers_OE_legs,
                     'momentum_EKF': momentum_EKF,
                     'momentum_OGE': momentum_OGE,
                     'momentum_OE': momentum_OE,
                     'RMSE_momentum_EKF': RMSE_momentum_EKF,
                     'RMSE_momentum_OGE': RMSE_momentum_OGE,
                     'RMSE_momentum_OE': RMSE_momentum_OE,
                     'total_mass': total_mass,
                     'linear_momentum_EKF': linear_momentum_kalman,
                     'linear_momentum_OGE': linear_momentum_optimal_gravity,
                     'linear_momentum_OE': linear_momentum,
                     'RMSE_linear_momentum_EKF': RMSE_linear_momentum_EKF,
                     'RMSE_linear_momentum_OGE': RMSE_linear_momentum_OGE,
                     'RMSE_linear_momentum_OE': RMSE_linear_momentum_OE,
                     'EKF_lm_gravity': slope_lm_kalman / total_mass / (duration / adjusted_number_shooting_points),
                     'OGE_lm_gravity': slope_lm_optimal_gravity / total_mass / (duration / adjusted_number_shooting_points),
                     'OE_lm_gravity': slope_lm / total_mass / (duration / adjusted_number_shooting_points),
                     'rank': rank, '%_full_rank': rank.count(biorbd_model.nbQ()) / adjusted_number_shooting_points * 100, },
                      handle, protocol=3)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)


if __name__ == "__main__":
    subjects_trials = [('DoCi', '822', 100), ('DoCi', '44_1', 100), ('DoCi', '44_2', 100), ('DoCi', '44_3', 100),
                       ('BeLa', '44_1', 100), ('BeLa', '44_2', 80), ('BeLa', '44_3', 100),
                       ('GuSe', '44_2', 80), ('GuSe', '44_3', 100), ('GuSe', '44_4', 100),
                       ('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100), ('SaMi', '821_contact_3', 100), ('SaMi', '822_contact_1', 100),
                       ('SaMi', '821_seul_1', 100), ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_3', 100), ('SaMi', '821_seul_4', 100), ('SaMi', '821_seul_5', 100),
                       ('SaMi', '821_822_2', 100), ('SaMi', '821_822_3', 100),
                       ('JeCh', '833_1', 100), ('JeCh', '833_2', 100), ('JeCh', '833_3', 100), ('JeCh', '833_4', 100), ('JeCh', '833_5', 100),
                      ]

    for subject, trial, number_shooting_points in subjects_trials:
        stats(subject, trial, number_shooting_points)

    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    # number_shooting_points = 100
    # trial = '821_822_3'
    #
    # stats(subject, trial, number_shooting_points)

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
    trial = '833_5'

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
    am = biorbd.to_casadi_func("am", biorbd_model.CalcAngularMomentum, q, qdot, qddot, True)
    fd = biorbd.to_casadi_func("fd", biorbd_model.ForwardDynamics, q, qdot, tau)
    mcm = biorbd.to_casadi_func("fd", biorbd_model.mass)
    vcm = biorbd.to_casadi_func("fd", biorbd_model.CoMdot, q, qdot)


    # --- Load --- #
    load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_noOGE'
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

    # rotating_gravity(biorbd_model, angle.squeeze())
    markers = states_to_markers(biorbd_model, ocp, states)
    markers_optimal_gravity = states_to_markers(biorbd_model, ocp, states_optimal_gravity)
    markers_kalman = states_to_markers(biorbd_model, ocp, states_kalman)
    markers_kalman_biorbd = states_to_markers(biorbd_model, ocp, states_kalman_biorbd)

    # --- Simulate --- #

    sol_simulate_OGE = Simulate.from_data(ocp, [states_optimal_gravity, controls_optimal_gravity], single_shoot=False)
    sol_simulate_OE = Simulate.from_data(ocp, [states, controls], single_shoot=False)
    # ShowResult(ocp, sol_simulate_OE).graphs()

    states_OGE_sim, controls_OGE_sim = Data.get_data(ocp, sol_simulate_OGE, integrate=True)
    states_OE_sim, controls_OE_sim = Data.get_data(ocp, sol_simulate_OE, integrate=True)

    # qddot_OGE_sim = fd(states_OGE_sim['q'], states_OGE_sim['q_dot'], controls_OGE_sim['tau'])
    # qddot_OE_sim = fd(states_OE_sim['q'], states_OE_sim['q_dot'], controls_OE_sim['tau'])
    qddot_OGE_sim = fd(states_OGE_sim['q'], states_OGE_sim['q_dot'], np.repeat(controls_OGE_sim['tau'], 2, axis=1)[:, :-1])
    qddot_OE_sim = fd(states_OE_sim['q'], states_OE_sim['q_dot'], np.repeat(controls_OE_sim['tau'], 2, axis=1)[:, :-1])

    sim_step_size = adjusted_number_shooting_points/(states_OE_sim['q'].shape[1]-1)

    # --- Stats --- #
    #OE
    average_distance_between_markers = (
            np.nanmean([np.sqrt(np.sum((markers[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)
    sd_distance_between_markers = (
            np.nanstd([np.sqrt(np.sum((markers[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)

    # OG
    average_distance_between_markers_optimal_gravity = (
            np.nanmean([np.sqrt(np.sum((markers_optimal_gravity[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)
    sd_distance_between_markers_optimal_gravity = (
            np.nanstd([np.sqrt(np.sum((markers_optimal_gravity[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)

    # EKF
    average_distance_between_markers_kalman = (
            np.nanmean([np.sqrt(np.sum((markers_kalman[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)
    sd_distance_between_markers_kalman = (
            np.nanstd([np.sqrt(np.sum((markers_kalman[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)

    # EKF biorbd
    average_distance_between_markers_kalman_biorbd = (
            np.nanmean([np.sqrt(np.sum((markers_kalman_biorbd[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)
    sd_distance_between_markers_kalman_biorbd = (
            np.nanstd([np.sqrt(np.sum((markers_kalman_biorbd[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)

    print('Number of shooting points: ', adjusted_number_shooting_points)
    print('Average marker error')
    print('Kalman: ', average_distance_between_markers_kalman, u"\u00B1", sd_distance_between_markers_kalman)
    print('Kalman biorbd: ', average_distance_between_markers_kalman_biorbd, u"\u00B1", sd_distance_between_markers_kalman_biorbd)
    print('Optimal gravity: ', average_distance_between_markers_optimal_gravity, u"\u00B1", sd_distance_between_markers_optimal_gravity)
    print('Estimation: ', average_distance_between_markers, u"\u00B1", sd_distance_between_markers)

    average_difference_between_Q_OG = np.sqrt(np.nanmean((states_kalman['q'] - states_optimal_gravity['q']) ** 2, 1))
    sd_difference_between_Q_OG = np.nanstd((states_kalman['q'] - states_optimal_gravity['q']), 1)
    average_difference_between_Q_OE = np.sqrt(np.nanmean((states_kalman['q'] - states['q']) ** 2, 1))
    sd_difference_between_Q_OE = np.nanstd((states_kalman['q'] - states['q']), 1)

    average_difference_between_Qd_OG = np.sqrt(np.nanmean((states_kalman['q_dot'] - states_optimal_gravity['q_dot']) ** 2, 1))
    sd_difference_between_Qd_OG = np.nanstd((states_kalman['q_dot'] - states_optimal_gravity['q_dot']), 1)
    average_difference_between_Qd_OE = np.sqrt(np.nanmean((states_kalman['q_dot'] - states['q_dot']) ** 2, 1))
    sd_difference_between_Qd_OE = np.nanstd((states_kalman['q_dot'] - states['q_dot']), 1)

    # print('Average state error vs Kalman')
    # print('Optimal gravity')
    # print('Q: ', average_difference_between_Q_OG, u"\u00B1", sd_difference_between_Q_OG)
    # print('Qd: ', average_difference_between_Qd_OG, u"\u00B1", sd_difference_between_Qd_OG)
    # print('Estimation')
    # print('Q: ', average_difference_between_Q_OE, u"\u00B1", sd_difference_between_Q_OE)
    # print('Qd: ', average_difference_between_Qd_OE, u"\u00B1", sd_difference_between_Qd_OE)

    momentum_OE = am(states['q'], states['q_dot'], qddot).full()
    momentum_OGE = am(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], qddot_optimal_gravity).full()
    momentum_EKF = am(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()
    # momentum_sim_OGE = am(states_OGE_sim['q'], states_OGE_sim['q_dot'], qddot_OGE_sim).full()
    # momentum_sim_OE = am(states_OE_sim['q'], states_OE_sim['q_dot'], qddot_OE_sim).full()

    median_OE = np.median(momentum_OE, axis=1)
    median_OGE = np.median(momentum_OGE, axis=1)
    median_EKF = np.median(momentum_EKF, axis=1)

    mad_OE = np.median(np.abs(momentum_OE - median_OE[:, np.newaxis]), axis=1)
    mad_OGE = np.median(np.abs(momentum_OGE - median_OGE[:, np.newaxis]), axis=1)
    mad_EKF = np.median(np.abs(momentum_EKF - median_EKF[:, np.newaxis]), axis=1)

    mean_OE = np.mean(momentum_OE, axis=1)[:, np.newaxis]
    mean_OGE = np.mean(momentum_OGE, axis=1)[:, np.newaxis]
    mean_EKF = np.mean(momentum_EKF, axis=1)[:, np.newaxis]

    total_mass = mcm()['o0'].full()
    linear_momentum = total_mass * vcm(states['q'], states['q_dot']).full()
    linear_momentum_optimal_gravity = total_mass * vcm(states_optimal_gravity['q'], states_optimal_gravity['q_dot']).full()
    linear_momentum_kalman = total_mass * vcm(q_kalman_biorbd, qdot_kalman_biorbd).full()
    # linear_momentum_sim_OGE = total_mass * vcm(states_OGE_sim['q'], states_OGE_sim['q_dot']).full()
    # linear_momentum_sim_OE = total_mass * vcm(states_OE_sim['q'], states_OE_sim['q_dot']).full()

    slope_lm, _ = np.polyfit(range(linear_momentum.shape[1]), linear_momentum.T, 1)/total_mass/(duration/adjusted_number_shooting_points)
    slope_lm_optimal_gravity, _ = np.polyfit(range(linear_momentum_optimal_gravity.shape[1]), linear_momentum_optimal_gravity.T, 1)/total_mass/(duration/adjusted_number_shooting_points)
    slope_lm_kalman, _ = np.polyfit(range(linear_momentum_kalman.shape[1]), linear_momentum_kalman.T, 1)/total_mass/(duration/adjusted_number_shooting_points)

    diff_lm = (linear_momentum[:, 1:] - linear_momentum[:, :-1])/total_mass/(duration/adjusted_number_shooting_points)
    diff_lm_optimal_gravity = (linear_momentum_optimal_gravity[:, 1:] - linear_momentum_optimal_gravity[:, :-1])/total_mass/(duration/adjusted_number_shooting_points)
    diff_lm_kalman = (linear_momentum_kalman[:, 1:] - linear_momentum_kalman[:, :-1])/total_mass/(duration/adjusted_number_shooting_points)


    # --- Plots --- #

    fig, axs = pyplot.subplots(2)
    axs[0].plot(momentum_OE[:, 1:].T, color='blue')
    axs[0].plot(momentum_OGE[:, 1:].T, color='orange', linestyle=':')
    axs[0].plot(momentum_EKF[:, 1:].T, color='green')
    # axs[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), momentum_sim_OGE.T, color='grey', linestyle=':')
    # axs[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), momentum_sim_OE.T, color='black', linestyle='--')

    # lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    # lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
    # lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    # pyplot.legend([lm_oe, lm_og, lm_kal], ['Kalman', 'OGE', 'OE'])
    # pyplot.title('Angular momentum of free fall movement')
    # pyplot.xlabel("Aerial time")
    axs[0].set_ylabel(r"$\mathregular{kg \cdot m^2/s}$")
    axs[0].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].tick_params(axis="y", direction='in')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)

    axs[0].annotate('x', (momentum_OE.shape[1] - 1, momentum_OE.T[-1, 0]), textcoords="offset points", xytext=(2, 0), ha='center')
    axs[0].annotate('y', (momentum_OE.shape[1] - 1, momentum_OE.T[-1, 1]), textcoords="offset points", xytext=(2, 0), ha='center')
    axs[0].annotate('z', (momentum_OE.shape[1] - 1, momentum_OE.T[-1, 2]), textcoords="offset points", xytext=(2, 0), ha='center')

    box_text = (
            'EKF MAD: x: ' + np.format_float_scientific(mad_EKF[0], precision=1, exp_digits=2) +
                   ', y: ' + np.format_float_scientific(mad_EKF[1], precision=1, exp_digits=2) +
                   ', z: ' + np.format_float_scientific(mad_EKF[2], precision=1, exp_digits=2) + '\n' +
            'OGE MAD: x: ' + np.format_float_scientific(mad_OGE[0], precision=1, exp_digits=2) +
                   ', y: ' + np.format_float_scientific(mad_OGE[1], precision=1, exp_digits=2) +
                   ', z: ' + np.format_float_scientific(mad_OGE[2], precision=1, exp_digits=2) + '\n' +
            'OE MAD: x: ' + np.format_float_scientific(mad_OE[0], precision=1, exp_digits=2) +
                  ', y: ' + np.format_float_scientific(mad_OE[1], precision=1, exp_digits=2) +
                  ', z: ' + np.format_float_scientific(mad_OE[2], precision=1, exp_digits=2)
    )
    text_box = AnchoredText(box_text, frameon=True, loc=5, pad=0.5, prop=dict(fontsize=9))
    pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
    axs[0].add_artist(text_box)

    axs[1].plot(linear_momentum[:, 1:].T, color='blue')
    axs[1].plot(linear_momentum_optimal_gravity[:, 1:].T, color='orange', linestyle=':')
    axs[1].plot(linear_momentum_kalman[:, 1:].T, color='green')
    # axs[1].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), linear_momentum_sim_OGE.T, color='grey', linestyle=':')
    # axs[1].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), linear_momentum_sim_OE.T, color='black', linestyle='--')

    lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
    lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    fig.legend([lm_kal, lm_og, lm_oe], ['EKF', 'OGE', 'OE'])
    # pyplot.title('Linear momentum of free fall movement')
    axs[1].set_xlabel("Aerial time")
    axs[1].set_ylabel(r"$\mathregular{kg \cdot m/s}$")
    axs[1].tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
    axs[1].tick_params(axis="y", direction='in')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

    axs[1].annotate('x', (linear_momentum.shape[1] - 1, linear_momentum.T[-1, 0]), textcoords="offset points", xytext=(2, -2), ha='center')
    axs[1].annotate('y', (linear_momentum.shape[1] - 1, linear_momentum.T[-1, 1]), textcoords="offset points", xytext=(2, 2), ha='center')
    axs[1].annotate('z', (linear_momentum.shape[1] - 1, linear_momentum.T[-1, 2]), textcoords="offset points", xytext=(2, 0), ha='center')

    box_text = (
            'EKF $\mathregular{G_{norm}}$: ' + f"{np.linalg.norm(slope_lm_kalman):.4f}" + '\n' +
            'OGE $\mathregular{G_{norm}}$: ' + f"{np.linalg.norm(slope_lm_optimal_gravity):.4f}" + '\n' +
            'OE $\mathregular{G_{norm}}$: ' + f"{np.linalg.norm(slope_lm):.4f}"
    )
    text_box = AnchoredText(box_text, frameon=True, loc=1, pad=0.5, prop=dict(fontsize=9))
    pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
    axs[1].add_artist(text_box)

    save_path = 'Solutions/'
    save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + "_momentum_N" + str(adjusted_number_shooting_points) + '.png'
    # pyplot.savefig(save_name)

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
    for idx_dof, dof in enumerate(dofs):
        fig, axs = pyplot.subplots(2)
        axs[0].plot(states_kalman['q'][dof, :].T, color='blue')
        axs[0].plot(states_optimal_gravity['q'][dof, :].T, color='red')
        axs[0].plot(states['q'][dof, :].T, color='green')
        # axs[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), states_OGE_sim['q'][dof, :].T, color='grey', linestyle=':')
        # axs[0].plot(np.arange(0, adjusted_number_shooting_points+sim_step_size, sim_step_size), states_OE_sim['q'][dof, :].T, color='black', linestyle='--')

        axs[1].plot(controls_kalman['tau'][dof, :].T, color='blue')
        axs[1].plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
        axs[1].plot(controls['tau'][dof, :].T, color='green')

        # fig = pyplot.figure()
        # pyplot.plot(qdot_ref_matlab[dof, :].T, color='blue')
        # pyplot.plot(qdot_ref_biorbd[dof, :].T, color='red')
        #
        # fig = pyplot.figure()
        # pyplot.plot(qddot_ref_matlab[dof, :].T, color='blue')
        # pyplot.plot(qddot_ref_biorbd[dof, :].T, color='red')

        fig.suptitle(dofs_name[idx_dof])
        lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
        lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        axs[0].legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])

        save_path = 'Solutions/'
        save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_' + dofs_name[idx_dof] + '.png'
        # pyplot.savefig(save_name)

    pyplot.show()

    # print('Angular momentum OE_noOGE: ', mean_OE.squeeze())
    # print('Linear momentum OE_noOGE: ', slope_lm)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
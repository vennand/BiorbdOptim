import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function
import pickle
from scipy.io import loadmat
import os
import sys
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers

from bioptim import (
    OptimalControlProgram,
    Simulate,
    Problem,
    Bounds,
    InitialGuess,
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


def states_to_markers(biorbd_model, states):
    q = states['q']
    n_q = biorbd_model.nbQ()
    n_mark = biorbd_model.nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("q", n_q, 1)
    # markers_func = Function(
    #     "markers_kyn", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    # ).expand()
    markers_func = biorbd.to_casadi_func("markers", biorbd_model.markers, symbolic_states)
    for i in range(n_frames):
        # markers[:, :, i] = markers_func(q[:, i])
        markers[:, :, i] = markers_func(q[:, i]).full()

    return markers


def dynamics(biorbd_model, q_ref, qd_ref, qdd_ref=None, tau_ref=None):
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    dynamics_dict = dict()

    if qdd_ref is not None:
        id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)
        am = biorbd.to_casadi_func("am", biorbd_model.CalcAngularMomentum, q, qdot, qddot, True)

        dynamics_dict['inverse_dynamics'] = {'tau': id(q_ref, qd_ref, qdd_ref)[:, :-1]}
        dynamics_dict['momentum'] = {'ret': am(q_ref, qd_ref, qdd_ref)}
    if tau_ref is not None:
        fd = biorbd.to_casadi_func("fd", biorbd_model.ForwardDynamics, q, qdot, tau)

        dynamics_dict['forward_dynamics'] = {'qdd': fd(q_ref, qd_ref, tau_ref)}

    return dynamics_dict


if __name__ == "__main__":
    subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '44_1'

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

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    # --- Functions --- #
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)
    am = biorbd.to_casadi_func("am", biorbd_model.CalcAngularMomentum, q, qdot, True)
    fd = biorbd.to_casadi_func("fd", biorbd_model.ForwardDynamics, q, qdot, tau)
    mcm = biorbd.to_casadi_func("fd", biorbd_model.mass)
    vcm = biorbd.to_casadi_func("fd", biorbd_model.CoMdot, q, qdot)


    # --- Load --- #
    load_path = 'Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
    ocp, sol = OptimalControlProgram.load(load_name + ".bo")

    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    markers_mocap = data['mocap']
    frames = data['frames']
    step_size = data['step_size']

    q_kalman = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size]
    qdot_kalman = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
    qddot_kalman = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]

    states_kalman = {'q': q_kalman, 'q_dot': qdot_kalman}
    controls_kalman = {'tau': id(q_kalman, qdot_kalman, qddot_kalman)}

    load_path = '/home/andre/bioptim/examples/optimal_gravity_ocp/Solutions/'
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
    # qddot = fd(states['q'], states['q_dot'], controls['tau'])

    optimal_gravity_filename = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
    # ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename + '.bo')
    # states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)

    with open(optimal_gravity_filename + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    # states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(data, get_parameters=True)
    states_optimal_gravity = data['states']
    controls_optimal_gravity = data['controls']
    params_optimal_gravity = data['params']


    angle = params_optimal_gravity["gravity_angle"].squeeze()
    # qddot_optimal_gravity = fd(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], controls_optimal_gravity['tau'])

    rotating_gravity(biorbd_model, angle.squeeze())
    markers = states_to_markers(biorbd_model, states)
    markers_optimal_gravity = states_to_markers(biorbd_model, states_optimal_gravity)
    # markers_optimal_gravity_EndChainMarkers = states_to_markers(biorbd_model, states_optimal_gravity_EndChainMarkers)
    markers_kalman = states_to_markers(biorbd_model, states_kalman)
    markers_kalman_biorbd = states_to_markers(biorbd_model, states_kalman_biorbd)

    # --- Simulate --- #

    # sol_simulate_OGE = Simulate.from_data(ocp, [states_optimal_gravity, controls_optimal_gravity], single_shoot=False)
    # sol_simulate_OE = Simulate.from_data(ocp, [states, controls], single_shoot=False)
    # ShowResult(ocp, sol_simulate_OGE).graphs()

    # states_OGE_sim, controls_OGE_sim = Data.get_data(ocp, sol_simulate_OGE)#, integrate=True)
    # states_OE_sim, controls_OE_sim = Data.get_data(ocp, sol_simulate_OE, integrate=True)

    # qddot_OGE_sim = fd(states_OGE_sim['q'], states_OGE_sim['q_dot'], controls_OGE_sim['tau'])
    # qddot_OE_sim = fd(states_OE_sim['q'], states_OE_sim['q_dot'], controls_OE_sim['tau'])
    # qddot_OGE_sim = fd(states_OGE_sim['q'], states_OGE_sim['q_dot'],np.repeat(controls_OGE_sim['tau'], 2, axis=1)[:, :-1])
    # # qddot_OE_sim = fd(states_OE_sim['q'], states_OE_sim['q_dot'], np.repeat(controls_OE_sim['tau'], 2, axis=1)[:, :-1])

    # sim_step_size = adjusted_number_shooting_points / (states_OE_sim['q'].shape[1] - 1)
    # sim_step_size = adjusted_number_shooting_points / (states_OGE_sim['q'].shape[1] - 1)

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

    momentum = am(states['q'], states['q_dot'])
    momentum_optimal_gravity = am(states_optimal_gravity['q'], states_optimal_gravity['q_dot'])
    momentum_kalman = am(q_kalman, qdot_kalman)

    total_mass = mcm()['o0'].full()
    linear_momentum = total_mass * vcm(states['q'], states['q_dot'])
    linear_momentum_optimal_gravity = total_mass * vcm(states_optimal_gravity['q'], states_optimal_gravity['q_dot'])
    linear_momentum_kalman = total_mass * vcm(q_kalman, qdot_kalman)

    slope_lm, _ = np.polyfit(range(linear_momentum.shape[1]), linear_momentum.T, 1)/total_mass/(len(frames)/200/number_shooting_points)
    slope_lm_optimal_gravity, _ = np.polyfit(range(linear_momentum_optimal_gravity.shape[1]), linear_momentum_optimal_gravity.T, 1)/total_mass/(len(frames)/200/number_shooting_points)
    slope_lm_kalman, _ = np.polyfit(range(linear_momentum_kalman.shape[1]), linear_momentum_kalman.T, 1)/total_mass/(len(frames)/200/number_shooting_points)


    # --- Plots --- #
    from matplotlib import pyplot
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredText
    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    lm_oe = pyplot.plot(momentum.T, color='blue')
    lm_og = pyplot.plot(momentum_optimal_gravity.T, color='orange', linestyle=':')
    lm_kal = pyplot.plot(momentum_kalman.T, color='green')

    lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
    lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    pyplot.legend([lm_oe, lm_og, lm_kal], ['Estimation', 'Optimal gravity', 'Kalman'])
    pyplot.title('Angular momentum of free fall movement')

    pyplot.annotate('x', (momentum.shape[1]-1, momentum.full().T[-1,0]), textcoords="offset points", xytext=(0,10), ha='center')
    pyplot.annotate('y', (momentum.shape[1]-1, momentum.full().T[-1,1]), textcoords="offset points", xytext=(0,10), ha='center')
    pyplot.annotate('z', (momentum.shape[1]-1, momentum.full().T[-1,2]), textcoords="offset points", xytext=(0,10), ha='center')

    # pyplot.savefig('Angular_momentum_N' + str(number_shooting_points) + '.png')

    fig = pyplot.figure()
    pyplot.plot(linear_momentum.T, color='blue')
    pyplot.plot(linear_momentum_optimal_gravity.T, color='orange', linestyle=':')
    pyplot.plot(linear_momentum_kalman.T, color='green')

    lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
    lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    pyplot.legend([lm_oe, lm_og, lm_kal], ['OE', 'OGE', 'Kalman'])
    pyplot.title('Linear momentum of free fall movement')

    pyplot.annotate('x', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 0]), textcoords="offset points", xytext=(0, 10), ha='center')
    pyplot.annotate('y', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
    pyplot.annotate('z', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 2]), textcoords="offset points", xytext=(0, 10), ha='center')

    box_text = (
            'Kalman gravity norm by linear regression: ' + f"{np.linalg.norm(slope_lm_kalman):.4f}" + '\n'
            'OG gravity norm: ' + f"{np.linalg.norm(slope_lm_optimal_gravity):.4f}" + '\n'
            'OE gravity norm: ' + f"{np.linalg.norm(slope_lm):.4f}"
    )
    text_box = AnchoredText(box_text, frameon=True, loc=3, pad=0.5)
    pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
    pyplot.gca().add_artist(text_box)

    # pyplot.savefig('Linear_momentum_N' + str(number_shooting_points) + '.png')

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
    # dofs = [range(0, 6), range(6, 9), range(9, 12),
    #         range(12, 14), range(14, 17), range(17, 19), range(19, 21)]
    for idx_dof, dof in enumerate(dofs):
        fig = pyplot.figure()
        pyplot.plot(states_kalman['q'][dof, :].T, color='blue')
        pyplot.plot(states_optimal_gravity['q'][dof, :].T, color='red')
        pyplot.plot(states['q'][dof, :].T, color='green')
        # pyplot.plot(states_OGE_sim['q'][dof, :].T, color='orange', linestyle=':')

        fig = pyplot.figure()
        pyplot.plot(states_optimal_gravity['q_dot'][dof, :].T, color='blue')
        # pyplot.plot(states_OGE_sim['q_dot'][dof, :].T, color='orange', linestyle=':')
        pyplot.plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
        #
        # fig = pyplot.figure()
        # pyplot.plot(qddot_ref_matlab[dof, :].T, color='blue')
        # pyplot.plot(qddot_ref_biorbd[dof, :].T, color='red')

        pyplot.title(dofs_name[idx_dof])
        lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
        lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        pyplot.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])

    pyplot.show()

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
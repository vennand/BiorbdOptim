import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function
import pickle
from scipy.io import loadmat

from biorbd_optim import (
    OptimalControlProgram,
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
    biorbd_model = biorbd.Model("DoCi.s2mMod")
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    number_shooting_points = 50

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
    load_name = "Do_822_contact_2_optimal_estimation_N" + str(number_shooting_points)
    load_ocp_sol_name = load_name + ".bo"
    ocp, sol = OptimalControlProgram.load(load_ocp_sol_name)

    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    markers_mocap = data['mocap']
    frames = data['frames']
    step_size = data['step_size']

    optimal_gravity_filename = "../optimal_gravity_ocp/Solutions/DoCi/Do_822_contact_2_optimal_gravity_N" + str(number_shooting_points) + ".bo"
    optimal_gravity_EndChainMarkers_filename = "../optimal_gravity_ocp/Solutions/DoCi/Do_822_contact_2_optimal_gravity_N" + str(number_shooting_points) + "_EndChainMarkers.bo"

    q_kalman = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat')['Q2'][:, frames.start:frames.stop:step_size]
    qdot_kalman = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat')['V2'][:, frames.start:frames.stop:step_size]
    qddot_kalman = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat')['A2'][:, frames.start:frames.stop:step_size]

    states_kalman = {'q': q_kalman, 'q_dot': qdot_kalman}
    controls_kalman = {'tau': id(q_kalman, qdot_kalman, qddot_kalman)}

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)
    qddot = fd(states['q'], states['q_dot'], controls['tau'])

    ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)
    angle = params_optimal_gravity["gravity_angle"].squeeze()
    qddot_optimal_gravity = fd(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], controls_optimal_gravity['tau'])

    ocp_optimal_gravity_EndChainMarkers, sol_optimal_gravity_EndChainMarkers = OptimalControlProgram.load(optimal_gravity_EndChainMarkers_filename)
    states_optimal_gravity_EndChainMarkers, controls_optimal_gravity_EndChainMarkers, params_optimal_gravity_EndChainMarkers = Data.get_data(ocp_optimal_gravity_EndChainMarkers, sol_optimal_gravity_EndChainMarkers, get_parameters=True)
    angle_EndChainMarkers = params_optimal_gravity_EndChainMarkers["gravity_angle"].squeeze()
    qddot_optimal_gravity_EndChainMarkers = fd(states_optimal_gravity_EndChainMarkers['q'], states_optimal_gravity_EndChainMarkers['q_dot'], controls_optimal_gravity_EndChainMarkers['tau'])

    rotating_gravity(biorbd_model, angle_EndChainMarkers.squeeze())
    markers = states_to_markers(biorbd_model, ocp, states)
    markers_optimal_gravity = states_to_markers(biorbd_model, ocp, states_optimal_gravity)
    markers_optimal_gravity_EndChainMarkers = states_to_markers(biorbd_model, ocp, states_optimal_gravity_EndChainMarkers)
    markers_kalman = states_to_markers(biorbd_model, ocp, states_kalman)

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

    # OG + ECM
    average_distance_between_markers_optimal_gravity_EndChainMarkers = (
            np.nanmean([np.sqrt(np.sum((markers_optimal_gravity_EndChainMarkers[:, i, j] - markers_mocap[:, i, j]) ** 2))
                        for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])]) * 1000)
    sd_distance_between_markers_optimal_gravity_EndChainMarkers = (
            np.nanstd([np.sqrt(np.sum((markers_optimal_gravity_EndChainMarkers[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)

    # EKF
    average_distance_between_markers_kalman = (
            np.nanmean([np.sqrt(np.sum((markers_kalman[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)
    sd_distance_between_markers_kalman = (
            np.nanstd([np.sqrt(np.sum((markers_kalman[:, i, j] - markers_mocap[:, i, j]) ** 2))
                for i in range(markers_mocap.shape[1]) for j in range(markers_mocap.shape[2])])*1000)

    print('Number of shooting points: ', number_shooting_points)
    print('Average marker error')
    print('Kalman: ', average_distance_between_markers_kalman, u"\u00B1", sd_distance_between_markers_kalman)
    print('Optimal gravity: ', average_distance_between_markers_optimal_gravity, u"\u00B1", sd_distance_between_markers_optimal_gravity)
    print('Optimal gravity with end chain markers: ', average_distance_between_markers_optimal_gravity_EndChainMarkers, u"\u00B1", sd_distance_between_markers_optimal_gravity_EndChainMarkers)
    print('Estimation: ', average_distance_between_markers, u"\u00B1", sd_distance_between_markers)

    average_difference_between_Q_OG = np.sqrt(np.nanmean((states_kalman['q'] - states_optimal_gravity['q']) ** 2, 1))
    sd_difference_between_Q_OG = np.nanstd((states_kalman['q'] - states_optimal_gravity['q']), 1)
    average_difference_between_Q_OE = np.sqrt(np.nanmean((states_kalman['q'] - states['q']) ** 2, 1))
    sd_difference_between_Q_OG = np.nanstd((states_kalman['q'] - states['q']), 1)

    average_difference_between_Qd_OG = np.sqrt(np.nanmean((states_kalman['q_dot'] - states_optimal_gravity['q_dot']) ** 2, 1))
    sd_difference_between_Q_OG = np.nanstd((states_kalman['q_dot'] - states_optimal_gravity['q_dot']), 1)
    average_difference_between_Qd_OE = np.sqrt(np.nanmean((states_kalman['q_dot'] - states['q_dot']) ** 2, 1))
    sd_difference_between_Q_OG = np.nanstd((states_kalman['q_dot'] - states['q_dot']), 1)

    momentum = am(states['q'], states['q_dot'], qddot)
    momentum_optimal_gravity = am(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], qddot_optimal_gravity)
    momentum_optimal_gravity_EndChainMarkers = am(states_optimal_gravity_EndChainMarkers['q'], states_optimal_gravity_EndChainMarkers['q_dot'], qddot_optimal_gravity_EndChainMarkers)
    momentum_kalman = am(q_kalman, qdot_kalman, qddot_kalman)

    total_mass = mcm()['o0'].full()
    linear_momentum = total_mass * vcm(states['q'], states['q_dot'])
    linear_momentum_optimal_gravity = total_mass * vcm(states_optimal_gravity['q'], states_optimal_gravity['q_dot'])
    linear_momentum_optimal_gravity_EndChainMarkers = total_mass * vcm(states_optimal_gravity_EndChainMarkers['q'], states_optimal_gravity_EndChainMarkers['q_dot'])
    linear_momentum_kalman = total_mass * vcm(q_kalman, qdot_kalman)

    slope_lm, _ = np.polyfit(range(linear_momentum.shape[1]), linear_momentum.T, 1)/total_mass/(len(frames)/200/number_shooting_points)
    slope_lm_optimal_gravity, _ = np.polyfit(range(linear_momentum_optimal_gravity.shape[1]), linear_momentum_optimal_gravity.T, 1)/total_mass/(len(frames)/200/number_shooting_points)
    slope_lm_optimal_gravity_EndChainMarkers, _ = np.polyfit(range(linear_momentum_optimal_gravity_EndChainMarkers.shape[1]), linear_momentum_optimal_gravity_EndChainMarkers.T, 1) / total_mass / (len(frames) / 200 / number_shooting_points)
    slope_lm_kalman, _ = np.polyfit(range(linear_momentum_kalman.shape[1]), linear_momentum_kalman.T, 1)/total_mass/(len(frames)/200/number_shooting_points)


    # --- Plots --- #
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D

    # fig = pyplot.figure()
    # ax = Axes3D(fig)
    #
    # ax.scatter(markers[0, :, 0], markers[1, :, 0], markers[2, :, 0])
    # ax.scatter(markers_mocap[0, :, 0], markers_mocap[1, :, 0], markers_mocap[2, :, 0])

    # for dof in range(controls['tau'].shape[0]):
    for dof in range(6,9):
        fig = pyplot.figure()
        pyplot.plot(controls['tau'][dof, :].T, color='blue')
        pyplot.plot(controls_optimal_gravity['tau'][dof, :].T, color='orange')
        pyplot.plot(controls_optimal_gravity_EndChainMarkers['tau'][dof, :].T, color='red')
        pyplot.plot(controls_kalman['tau'][dof, :].T, color='green')
        pyplot.title('DoF ' + str(dof+1))
        pyplot.legend(['Estimation', 'Optimal gravity', 'Optimal gravity with end chain markers', 'Kalman'])
        pyplot.savefig('Thorax_control_N' + str(number_shooting_points) + '_DoF' + str(dof+1) + '.png')

    fig = pyplot.figure()
    pyplot.plot(momentum.T, color='blue')
    pyplot.plot(momentum_optimal_gravity.T, color='orange', linestyle=':')
    pyplot.plot(momentum_optimal_gravity_EndChainMarkers.T, color='red', linestyle=':')
    pyplot.plot(momentum_kalman.T, color='green')
    pyplot.legend(['Estimation', 'Optimal gravity', 'Optimal gravity with end chain markers', 'Kalman'])
    pyplot.title('Angular momentum of free fall movement')

    pyplot.annotate('x', (momentum.shape[1]-1, momentum.full().T[-1,0]), textcoords="offset points", xytext=(0,10), ha='center')
    pyplot.annotate('y', (momentum.shape[1]-1, momentum.full().T[-1,1]), textcoords="offset points", xytext=(0,10), ha='center')
    pyplot.annotate('z', (momentum.shape[1]-1, momentum.full().T[-1,2]), textcoords="offset points", xytext=(0,10), ha='center')

    ax = pyplot.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('blue')
    leg.legendHandles[1].set_color('orange')
    leg.legendHandles[2].set_color('red')
    leg.legendHandles[3].set_color('green')

    # pyplot.savefig('Angular_momentum_N' + str(number_shooting_points) + '.png')

    fig = pyplot.figure()
    pyplot.plot(linear_momentum.T, color='blue')
    pyplot.plot(linear_momentum_optimal_gravity.T, color='orange', linestyle=':')
    pyplot.plot(linear_momentum_optimal_gravity_EndChainMarkers.T, color='red', linestyle=':')
    pyplot.plot(linear_momentum_kalman.T, color='green')
    pyplot.legend(['Estimation', 'Optimal gravity', 'Optimal gravity with end chain markers', 'Kalman'])
    pyplot.title('Linear momentum of free fall movement')

    pyplot.annotate('x', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 0]), textcoords="offset points", xytext=(0, 10), ha='center')
    pyplot.annotate('y', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
    pyplot.annotate('z', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 2]), textcoords="offset points", xytext=(0, 10), ha='center')

    ax = pyplot.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('blue')
    leg.legendHandles[1].set_color('orange')
    leg.legendHandles[2].set_color('red')
    leg.legendHandles[3].set_color('green')

    # pyplot.savefig('Linear_momentum_N' + str(number_shooting_points) + '.png')

    pyplot.show()

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=number_shooting_points)
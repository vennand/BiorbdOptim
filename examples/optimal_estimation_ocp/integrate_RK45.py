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


if __name__ == "__main__":
    subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '44_1'

    single_shoot = True

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

    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency
    dt = duration / adjusted_number_shooting_points

    # --- Load --- #
    load_path = 'Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
    ocp, sol = OptimalControlProgram.load(load_name + ".bo")
    states, controls = Data.get_data(ocp, sol)

    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    load_path = '/home/andre/bioptim/examples/optimal_gravity_ocp/Solutions/'
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

    rotating_gravity(biorbd_model, angle)

    load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)
    q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

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

    if single_shoot:
        def dyn_interface(t, x, u):
            return np.concatenate((x[nb_q:nb_q + nb_qdot], np.array(fd(x[:nb_q], x[nb_q:nb_q + nb_qdot], u)).squeeze()))

        states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
        controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)}

        # --- Simulate --- #

        from scipy.integrate import solve_ivp

        X = np.ndarray((nb_q + nb_qdot, adjusted_number_shooting_points + 1))

        x_init = np.concatenate((states_optimal_gravity['q'][:, 0], states_optimal_gravity['q_dot'][:, 0]))
        X[:, 0] = x_init
        for i, u in enumerate(controls_optimal_gravity['tau'][:, :-1].T):
            sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))

            x_init = sol["y"][:, -1]
            X[:, i+1] = x_init
    else:
        def dyn_interface(t, x, u):
            return np.concatenate((x[nb_q:nb_q + nb_qdot], np.array(fd(x[:nb_q], x[nb_q:nb_q + nb_qdot], u)).squeeze()))

        states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
        controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)}

        # --- Simulate --- #

        from scipy.integrate import solve_ivp

        X = np.ndarray((nb_q + nb_qdot, adjusted_number_shooting_points + 1))

        x = np.concatenate((states_optimal_gravity['q'], states_optimal_gravity['q_dot']))
        X[:, 0] = x[:, 0]
        for i, u in enumerate(controls_optimal_gravity['tau'][:, :-1].T):
            sol = solve_ivp(dyn_interface, (0, dt), x[:, i], method="RK45", args=(u,))

            X[:, i+1] = sol["y"][:, -1]

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
    momentum = am(states['q'], states['q_dot'])
    momentum_optimal_gravity = am(states_optimal_gravity['q'], states_optimal_gravity['q_dot'])
    momentum_kalman = am(q_kalman_biorbd, qdot_kalman_biorbd)

    total_mass = mcm()['o0'].full()
    linear_momentum = total_mass * vcm(states['q'], states['q_dot'])
    linear_momentum_optimal_gravity = total_mass * vcm(states_optimal_gravity['q'], states_optimal_gravity['q_dot'])
    linear_momentum_kalman = total_mass * vcm(q_kalman_biorbd, qdot_kalman_biorbd)

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

    pyplot.annotate('x', (momentum.shape[1] - 1, momentum.full().T[-1, 0]), textcoords="offset points", xytext=(0, 10),
                    ha='center')
    pyplot.annotate('y', (momentum.shape[1] - 1, momentum.full().T[-1, 1]), textcoords="offset points", xytext=(0, 10),
                    ha='center')
    pyplot.annotate('z', (momentum.shape[1] - 1, momentum.full().T[-1, 2]), textcoords="offset points", xytext=(0, 10),
                    ha='center')

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

    pyplot.annotate('x', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 0]), textcoords="offset points",
                    xytext=(0, 10), ha='center')
    pyplot.annotate('y', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 1]), textcoords="offset points",
                    xytext=(0, 10), ha='center')
    pyplot.annotate('z', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 2]), textcoords="offset points",
                    xytext=(0, 10), ha='center')

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
        pyplot.plot(states_kalman_biorbd['q'][dof, :].T, color='blue')
        pyplot.plot(states_optimal_gravity['q'][dof, :].T, color='red')
        pyplot.plot(X[dof, :].T, color='green', linestyle=':')
        # pyplot.plot(states_OGE_sim['q'][dof, :].T, color='orange', linestyle=':')

        # fig = pyplot.figure()
        # pyplot.plot(states_optimal_gravity['q_dot'][dof, :].T, color='blue')
        # # pyplot.plot(states_OGE_sim['q_dot'][dof, :].T, color='orange', linestyle=':')
        # pyplot.plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
        #
        # fig = pyplot.figure()
        # pyplot.plot(qddot_ref_matlab[dof, :].T, color='blue')
        # pyplot.plot(qddot_ref_biorbd[dof, :].T, color='red')

        pyplot.title(dofs_name[idx_dof])
        lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
        pyplot.legend([lm_kalman, lm_OGE], ['Kalman', 'OGE'])

    pyplot.show()

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
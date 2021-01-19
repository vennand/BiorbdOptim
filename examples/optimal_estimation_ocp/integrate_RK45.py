import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function
import pickle
from scipy.io import loadmat
from scipy.integrate import solve_ivp
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

    single_shoot = False

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

    c3d = ezc3d.c3d(c3d_path + c3d_name)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency
    dt = duration / adjusted_number_shooting_points

    # --- Load --- #
    load_path = '/home/andre/bioptim/examples/optimal_gravity_ocp/Solutions/'
    filename_RK4_B200 = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
    filename_RK20_B200 = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_RK20'
    filename_RK4_B50 = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_V50'
    filename_RK4_B200_MinTorqDiff_5 = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_MinTorqDiff-5'
    filename_RK4_B200_MinTorqDiff_6 = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_MinTorqDiff-6'

    with open(filename_RK4_B200 + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    states_RK4_B200 = data['states']
    controls_RK4_B200 = data['controls']
    angle_RK4_B200 = data['params']["gravity_angle"].squeeze()

    biorbd_model_RK4_B200 = biorbd.Model(model_path + model_name)
    biorbd_model_RK4_B200.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    nb_q = biorbd_model_RK4_B200.nbQ()
    nb_qdot = biorbd_model_RK4_B200.nbQdot()

    with open(filename_RK20_B200 + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    states_RK20_B200 = data['states']
    controls_RK20_B200 = data['controls']
    angle_RK20_B200 = data['params']["gravity_angle"].squeeze()

    biorbd_model_RK20_B200 = biorbd.Model(model_path + model_name)
    biorbd_model_RK20_B200.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    with open(filename_RK4_B50 + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    states_RK4_B50 = data['states']
    controls_RK4_B50 = data['controls']
    angle_RK4_B50 = data['params']["gravity_angle"].squeeze()

    biorbd_model_RK4_B50 = biorbd.Model(model_path + model_name)
    biorbd_model_RK4_B50.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    with open(filename_RK4_B200_MinTorqDiff_5 + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    states_RK4_B200_MinTorqDiff_5 = data['states']
    controls_RK4_B200_MinTorqDiff_5 = data['controls']
    angle_RK4_B200_MinTorqDiff_5 = data['params']["gravity_angle"].squeeze()

    biorbd_model_RK4_B200_MinTorqDiff_5 = biorbd.Model(model_path + model_name)
    biorbd_model_RK4_B200_MinTorqDiff_5.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    with open(filename_RK4_B200_MinTorqDiff_6 + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    states_RK4_B200_MinTorqDiff_6 = data['states']
    controls_RK4_B200_MinTorqDiff_6 = data['controls']
    angle_RK4_B200_MinTorqDiff_6 = data['params']["gravity_angle"].squeeze()

    biorbd_model_RK4_B200_MinTorqDiff_6 = biorbd.Model(model_path + model_name)
    biorbd_model_RK4_B200_MinTorqDiff_6.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    def integrate_RK45_and_calc_momentum(model, states, controls, angle, single_shoot=False):
        rotating_gravity(model, angle)

        # --- Functions --- #
        q = MX.sym("Q", model.nbQ(), 1)
        qdot = MX.sym("Qdot", model.nbQdot(), 1)
        tau = MX.sym("Tau", model.nbQddot(), 1)

        am = biorbd.to_casadi_func("am", model.CalcAngularMomentum, q, qdot, True)
        fd = biorbd.to_casadi_func("fd", model.ForwardDynamics, q, qdot, tau)

        def dyn_interface(t, x, u):
            return np.concatenate((x[nb_q:nb_q + nb_qdot], np.array(fd(x[:nb_q], x[nb_q:nb_q + nb_qdot], u)).squeeze()))

        # --- Simulate --- #
        X = np.ndarray((nb_q + nb_qdot, adjusted_number_shooting_points + 1))

        if single_shoot:
            x_init = np.concatenate((states['q'][:, 0], states['q_dot'][:, 0]))
            X[:, 0] = x_init
            for i, u in enumerate(controls['tau'][:, :-1].T):
                sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))

                x_init = sol["y"][:, -1]
                X[:, i+1] = x_init
        else:
            x = np.concatenate((states['q'], states['q_dot']))
            X[:, 0] = x[:, 0]
            for i, u in enumerate(controls['tau'][:, :-1].T):
                sol = solve_ivp(dyn_interface, (0, dt), x[:, i], method="RK45", args=(u,))

                X[:, i+1] = sol["y"][:, -1]

        # --- Stats --- #
        momentum = am(states['q'], states['q_dot'])

        return momentum, X


    momentum_RK4_B200, X_RK4_B200 = integrate_RK45_and_calc_momentum(biorbd_model_RK4_B200, states_RK4_B200, controls_RK4_B200, angle_RK4_B200)
    momentum_RK20_B200, X_RK20_B200 = integrate_RK45_and_calc_momentum(biorbd_model_RK20_B200, states_RK20_B200, controls_RK20_B200, angle_RK20_B200)
    momentum_RK4_B50, X_RK4_B50 = integrate_RK45_and_calc_momentum(biorbd_model_RK4_B50, states_RK4_B50, controls_RK4_B50, angle_RK4_B50)
    momentum_RK4_B200_MinTorqDiff_5, X_RK4_B200_MinTorqDiff_5 = integrate_RK45_and_calc_momentum(biorbd_model_RK4_B200_MinTorqDiff_5, states_RK4_B200_MinTorqDiff_5, controls_RK4_B200_MinTorqDiff_5, angle_RK4_B200_MinTorqDiff_5)
    momentum_RK4_B200_MinTorqDiff_6, X_RK4_B200_MinTorqDiff_6 = integrate_RK45_and_calc_momentum(biorbd_model_RK4_B200_MinTorqDiff_6, states_RK4_B200_MinTorqDiff_6, controls_RK4_B200_MinTorqDiff_6, angle_RK4_B200_MinTorqDiff_6)

    # --- Plots --- #
    from matplotlib import pyplot
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredText
    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    lm_RK4_B200 = pyplot.plot(momentum_RK4_B200.T, color='blue')
    lm_RK20_B200 = pyplot.plot(momentum_RK20_B200.T, color='orange')
    lm_RK4_B50 = pyplot.plot(momentum_RK4_B50.T, color='green')
    lm_RK4_B200_MinTorqDiff_5 = pyplot.plot(momentum_RK4_B200_MinTorqDiff_5.T, color='red')
    lm_RK4_B200_MinTorqDiff_6 = pyplot.plot(momentum_RK4_B200_MinTorqDiff_6.T, color='purple')

    lm_RK4_B200 = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_RK20_B200 = Line2D([0, 1], [0, 1], linestyle='-', color='orange')
    lm_RK4_B50 = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    lm_RK4_B200_MinTorqDiff_5 = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    lm_RK4_B200_MinTorqDiff_6 = Line2D([0, 1], [0, 1], linestyle='-', color='purple')
    pyplot.legend([lm_RK4_B200, lm_RK20_B200, lm_RK4_B50, lm_RK4_B200_MinTorqDiff_5, lm_RK4_B200_MinTorqDiff_6],
                  ['RK4 steps: 4, Bounds limit: 200',
                   'RK4 steps: 20, Bounds limit: 200',
                   'RK4 steps: 4, Bounds limit: 50',
                   'RK4 steps: 4, Bounds limit: 200, Min Torq Diff 1e-5',
                   'RK4 steps: 4, Bounds limit: 200, Min Torq Diff 1e-6'])
    pyplot.title('Angular momentum of free fall movement')

    pyplot.annotate('x', (momentum_RK4_B200.shape[1] - 1, momentum_RK4_B200.full().T[-1, 0]), textcoords="offset points", xytext=(0, 10),
                    ha='center')
    pyplot.annotate('y', (momentum_RK4_B200.shape[1] - 1, momentum_RK4_B200.full().T[-1, 1]), textcoords="offset points", xytext=(0, 10),
                    ha='center')
    pyplot.annotate('z', (momentum_RK4_B200.shape[1] - 1, momentum_RK4_B200.full().T[-1, 2]), textcoords="offset points", xytext=(0, 10),
                    ha='center')

    # pyplot.savefig('Angular_momentum_N' + str(number_shooting_points) + '.png')

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
        # pyplot.plot(states_RK4_B200['q'][dof, :].T, color='blue', linestyle=':')
        # pyplot.plot(states_RK20_B200['q'][dof, :].T, color='orange', linestyle=':')
        # pyplot.plot(states_RK4_B50['q'][dof, :].T, color='green', linestyle=':')
        # pyplot.plot(states_RK4_B200_MinTorqDiff_5['q'][dof, :].T, color='red', linestyle=':')
        # pyplot.plot(states_RK4_B200_MinTorqDiff_6['q'][dof, :].T, color='purple', linestyle=':')
        #
        # pyplot.plot(X_RK4_B200[dof, :].T, color='blue')
        # pyplot.plot(X_RK20_B200[dof, :].T, color='orange')
        # pyplot.plot(X_RK4_B50[dof, :].T, color='green')
        # pyplot.plot(X_RK4_B200_MinTorqDiff_5[dof, :].T, color='red')
        # pyplot.plot(X_RK4_B200_MinTorqDiff_6[dof, :].T, color='purple')

        # pyplot.plot(states_RK4_B200['q_dot'][dof, :].T, color='blue', linestyle=':')
        # pyplot.plot(states_RK20_B200['q_dot'][dof, :].T, color='orange', linestyle=':')
        # pyplot.plot(states_RK4_B50['q_dot'][dof, :].T, color='green', linestyle=':')
        # pyplot.plot(states_RK4_B200_MinTorqDiff_5['q_dot'][dof, :].T, color='red', linestyle=':')
        # pyplot.plot(states_RK4_B200_MinTorqDiff_6['q_dot'][dof, :].T, color='purple', linestyle=':')
        #
        # pyplot.plot(X_RK4_B200[nb_q:, :][dof, :].T, color='blue')
        # pyplot.plot(X_RK20_B200[nb_q:, :][dof, :].T, color='orange')
        # pyplot.plot(X_RK4_B50[nb_q:, :][dof, :].T, color='green')
        # pyplot.plot(X_RK4_B200_MinTorqDiff_5[nb_q:, :][dof, :].T, color='red')
        # pyplot.plot(X_RK4_B200_MinTorqDiff_6[nb_q:, :][dof, :].T, color='purple')

        pyplot.plot(controls_RK4_B200['tau'][dof, :].T, color='blue', linestyle=':')
        pyplot.plot(controls_RK20_B200['tau'][dof, :].T, color='orange', linestyle=':')
        pyplot.plot(controls_RK4_B50['tau'][dof, :].T, color='green', linestyle=':')
        pyplot.plot(controls_RK4_B200_MinTorqDiff_5['tau'][dof, :].T, color='red', linestyle=':')
        pyplot.plot(controls_RK4_B200_MinTorqDiff_6['tau'][dof, :].T, color='purple', linestyle=':')

        pyplot.title(dofs_name[idx_dof])
        lm_RK4_B200 = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_RK20_B200 = Line2D([0, 1], [0, 1], linestyle='-', color='orange')
        lm_RK4_B50 = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        lm_RK4_B200_MinTorqDiff_5 = Line2D([0, 1], [0, 1], linestyle='-', color='red')
        lm_RK4_B200_MinTorqDiff_6 = Line2D([0, 1], [0, 1], linestyle='-', color='purple')
        pyplot.legend([lm_RK4_B200, lm_RK20_B200, lm_RK4_B50, lm_RK4_B200_MinTorqDiff_5, lm_RK4_B200_MinTorqDiff_6],
                      ['RK4 steps: 4, Bounds limit: 200',
                       'RK4 steps: 20, Bounds limit: 200',
                       'RK4 steps: 4, Bounds limit: 50',
                       'RK4 steps: 4, Bounds limit: 200, Min Torq Diff 1e-5',
                       'RK4 steps: 4, Bounds limit: 200, Min Torq Diff 1e-6'])

    pyplot.show()

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
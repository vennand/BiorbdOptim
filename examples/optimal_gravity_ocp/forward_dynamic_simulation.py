import biorbd
import numpy as np
import ezc3d
from casadi import MX, DM, Function, vertcat
from scipy.integrate import solve_ivp
import pickle
from scipy.io import loadmat
import os
import sys

from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from x_bounds import x_bounds

from matplotlib import pyplot
from matplotlib.lines import Line2D

from bioptim import (
    OptimalControlProgram,
    Simulate,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    InitialConditionsList,
    InitialConditions,
    Problem,
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
    gravity.applyRT(biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


def simulate_data(biorbd_model, ocp, Tf, X0, U, N):
    ode = lambda t, x, u: ocp.nlp[0]['dynamics_func'](x, u, 0).toarray().squeeze()

    # Simulated data
    h = Tf / N
    X_ = np.zeros((biorbd_model.nbQ() + biorbd_model.nbQdot(), N+1))  # State trajectory
    sol_states = np.copy(X0)

    for n in range(N):
        sol = solve_ivp(ode, [0, h], X0, args=(U[:, n],))
        X_[:, n] = X0
        X0 = sol["y"][:, -1]
        sol_states = np.concatenate([sol_states, U[:, n], X0])
    X_[:, -1] = X0
    sol_states_dict = {'x': sol_states}

    return X_, sol_states_dict

def prepare_ocp(biorbd_model, final_time, number_shooting_points, q_ref, qdot_ref, tau_init, xmin, xmax):
    # --- Options --- #
    torque_min, torque_max = -300, 300
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(Bounds(min_bound=xmin, max_bound=xmax))


    # Initial guess
    X_init = InitialConditionsList()
    X_init.add(np.concatenate([q_ref, qdot_ref]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialConditionsList()
    U_init.add(tau_init, interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        nb_integration_steps=20,
        nb_threads=4,
    )

if __name__ == "__main__":
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_seul_4'

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
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Functions --- #
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    # tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    # --- Load data ---#
    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    optimal_gravity_filename = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_ultra_precise' + ".bo"

    q_kalman = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size]
    qdot_kalman = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
    qddot_kalman = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]

    states_kalman = {'q': q_kalman, 'q_dot': qdot_kalman}
    controls_kalman = {'tau': id(q_kalman, qdot_kalman, qddot_kalman)[:, :-1]}

    load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)
    q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

    states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
    controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)[:, :-1]}

    ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)
    angle = params_optimal_gravity["gravity_angle"].squeeze()
    print('Gravity angle', angle/np.pi*180)
    rotating_gravity(biorbd_model, angle.squeeze())

    xmin, xmax = x_bounds(biorbd_model)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        q_ref=states_optimal_gravity['q'], qdot_ref=states_optimal_gravity['q_dot'],
        tau_init=controls_optimal_gravity['tau'][:, :-1],
        xmin=xmin, xmax=xmax,
    )

    # --- Single shooting --- #
    sol_simulate_EKF_matlab = Simulate.from_data(ocp, [states_kalman, controls_kalman], single_shoot=True)
    sol_simulate_EKF_biorbd = Simulate.from_data(ocp, [states_kalman_biorbd, controls_kalman_biorbd], single_shoot=True)
    sol_simulate_OGE = Simulate.from_data(ocp, [states_optimal_gravity, controls_optimal_gravity], single_shoot=True)

    # sol_simulate_EKF_matlab['x'] = np.append(sol_simulate_EKF_matlab['x'], [0, 0])
    # sol_simulate_EKF_biorbd['x'] = np.append(sol_simulate_EKF_biorbd['x'], [0, 0])
    # sol_simulate_OGE['x'] = np.append(sol_simulate_OGE['x'], [0, 0])

    # X_OGE_sim_FD, sol_OGE_sim_FD = simulate_data(biorbd_model, ocp, duration,
    #                                              np.concatenate((states_optimal_gravity['q'][:, 0], states_optimal_gravity['q_dot'][:, 0])),
    #                                              controls_optimal_gravity['tau'], adjusted_number_shooting_points)
    # Q_OGE_sim_FD = X_OGE_sim_FD[: biorbd_model.nbQ(),:]

    states_kalman_sim, controls_kalman_sim = Data.get_data(ocp, sol_simulate_EKF_matlab)
    states_kalman_biorbd_sim, controls_kalman_biorbd_sim = Data.get_data(ocp, sol_simulate_EKF_biorbd)
    states_OGE_sim, controls_OGE_sim = Data.get_data(ocp, sol_simulate_OGE)

    diff_q_kalman = np.nanmean(np.absolute(states_kalman_sim['q'] - states_kalman['q']), axis=0)
    diff_q_kalman_biorbd = np.nanmean(np.absolute(states_kalman_biorbd_sim['q'] - states_kalman_biorbd['q']), axis=0)
    diff_q_OGE = np.nanmean(np.absolute(states_OGE_sim['q'] - states_optimal_gravity['q']), axis=0)
    # diff_q_OGE_FD = np.nanmean(np.absolute(Q_OGE_sim_FD - states_optimal_gravity['q']), axis=0)

    fig = pyplot.figure()
    pyplot.plot(diff_q_kalman, color='red')
    # pyplot.plot(diff_q_kalman_biorbd, color='blue')
    pyplot.plot(diff_q_OGE, color='green')
    fig = pyplot.figure()
    pyplot.plot(np.log(diff_q_kalman), color='red')
    # pyplot.plot(np.log(diff_q_kalman_biorbd), color='blue')
    pyplot.plot(np.log(diff_q_OGE), color='green')
    # pyplot.plot(diff_q_OGE_FD, color='orange')

    # pyplot.ylim(-1, 5)

    pyplot.show()

    # sol_simulate_EKF_matlab['x'][np.isnan(sol_simulate_EKF_matlab['x'])] = 0
    # sol_simulate_EKF_biorbd['x'][np.isnan(sol_simulate_EKF_biorbd['x'])] = 0
    sol_simulate_OGE['x'][np.isnan(sol_simulate_OGE['x'])] = 0
    ShowResult(ocp, sol_simulate_OGE).animate(nb_frames=adjusted_number_shooting_points)

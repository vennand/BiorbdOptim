import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import pickle
import time
from casadi import MX, Function
import os
import warnings
from load_data_filename import load_data_filename
from x_bounds import x_bounds
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    Bounds,
    InitialGuessList,
    InitialGuess,
    ShowResult,
    InterpolationType,
    Data,
    ParameterList,
    Solver,
    OdeSolver,
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


def inverse_dynamics(biorbd_model, q_ref, qd_ref, qdd_ref):
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    return id(q_ref, qd_ref, qdd_ref)[:, :-1]


def check_Kalman(q_ref):
    segments_with_pi_limits = [14, 16, 17, 18, 23, 25, 26, 27, 30, 33, 36, 39]
    hand_segments = [19, 20, 28, 29]
    bool = np.zeros(q_ref.shape)
    bool[4, :] = ((q_ref[4, :] / (np.pi / 2)).astype(int) != 0)
    for (i, j), q in np.ndenumerate(q_ref[6:, :]):
        if i+6 in segments_with_pi_limits:
            bool[i+6, j] = ((q / np.pi).astype(int) != 0)
        elif i+6 in hand_segments:
            bool[i+6, j] = ((q / (3*np.pi/2)).astype(int) != 0)
        else:
            bool[i+6, j] = ((q / (np.pi/2)).astype(int) != 0)
    states_idx_bool = bool.any(axis=1)

    states_idx_range_list = []
    start_index = 0
    broken_dofs = []
    for idx, bool_idx in enumerate(states_idx_bool):
        if bool_idx:
            stop_index = idx
            if idx != start_index:
                states_idx_range_list.append(range(start_index, stop_index))
            start_index = stop_index + 1
            broken_dofs.append(stop_index)
    if bool.shape[0] != start_index:
        states_idx_range_list.append(range(start_index, bool.shape[0]))
    return states_idx_range_list, broken_dofs


def choose_Kalman(q_ref_1, qdot_ref_1, qddot_ref_1, q_ref_2, qdot_ref_2, qddot_ref_2):
    _, broken_dofs_1 = check_Kalman(q_ref_1)
    _, broken_dofs_2 = check_Kalman(q_ref_2)

    q_ref_chosen = q_ref_1
    qdot_ref_chosen = qdot_ref_1
    qddot_ref_chosen = qddot_ref_1

    for dof in broken_dofs_1:
        if dof not in broken_dofs_2:
            q_ref_chosen[dof, :] = q_ref_2[dof, :]
            qdot_ref_chosen[dof, :] = qdot_ref_2[dof, :]
            qddot_ref_chosen[dof, :] = qddot_ref_2[dof, :]

    return q_ref_chosen, qdot_ref_chosen, qddot_ref_chosen


def shift_by_pi(q, error_margin):
    if ((np.pi)*(1-error_margin)) < np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q - np.pi
    elif ((np.pi)*(1-error_margin)) < -np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q + np.pi

    return q


def correct_Kalman(biorbd_model, q):
    error_margin = 0.35

    first_dof_segments_with_3DoFs = [6, 9, 14, 23, 30, 36]
    first_dof_segments_with_2DoFs = [12, 17, 19, 21, 26, 28, 34, 40]

    n_q = biorbd_model.nbQ()
    q[4, :] = q[4, :] - ((2 * np.pi) * (np.mean(q[4, :]) / (2 * np.pi)).astype(int))
    for dof in range(6, n_q):
        q[dof, :] = q[dof, :] - ((2*np.pi) * (np.mean(q[dof, :]) / (2*np.pi)).astype(int))
        if ((2*np.pi)*(1-error_margin)) < np.mean(q[dof, :]) < ((2*np.pi)*(1+error_margin)):
            q[dof, :] = q[dof, :] - (2*np.pi)
        elif ((2*np.pi)*(1-error_margin)) < -np.mean(q[dof, :]) < ((2*np.pi)*(1+error_margin)):
            q[dof, :] = q[dof, :] + (2*np.pi)

    if ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[4, :])) < ((np.pi) * (1 + error_margin)):
        q[3, :] = shift_by_pi(q[3, :], error_margin)
        q[4, :] = -shift_by_pi(q[4, :], error_margin)
        q[5, :] = shift_by_pi(q[5, :], error_margin)

    for dof in first_dof_segments_with_2DoFs:
        if ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof, :])) < ((np.pi) * (1 + error_margin)):
            q[dof, :] = shift_by_pi(q[dof, :], error_margin)
            q[dof+1, :] = -shift_by_pi(q[dof+1, :], error_margin)

    for dof in first_dof_segments_with_3DoFs:
        if (((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof, :])) < ((np.pi) * (1 + error_margin)) or
            ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof+1, :])) < ((np.pi) * (1 + error_margin)) or
            ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof+2, :])) < ((np.pi) * (1 + error_margin))):
            q[dof, :] = shift_by_pi(q[dof, :], error_margin)
            q[dof+1, :] = -shift_by_pi(q[dof+1, :], error_margin)
            q[dof+2, :] = shift_by_pi(q[dof+2, :], error_margin)

    return q


def prepare_ocp(biorbd_model, final_time, number_shooting_points, q_ref, qdot_ref, tau_init, xmin, xmax, min_g, max_g, use_ACADOS, markers_ref=None, markers_idx_ref=None, states_idx_ref=None, broken_dofs=None):
    # --- Options --- #
    torque_min, torque_max = -300, 300
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_ref, qdot_ref))
    objective_functions = ObjectiveList()
    if markers_idx_ref and markers_ref is not None:
        for markers_idx_range in markers_idx_ref:
            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=1, target=markers_ref[markers_idx_range, :],
                                    index=markers_idx_range)
    if states_idx_ref is not None:
        for states_idx_range in states_idx_ref:
            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, target=state_ref[states_idx_range, :],
                                    index=states_idx_range)
    else:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, target=state_ref[range(n_q), :],
             index=range(n_q))
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=1e-6, target=state_ref[range(n_q, n_q + n_qdot), :], index=range(n_q, n_q + n_qdot))
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1e-5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-7)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(min_bound=xmin, max_bound=xmax)


    # Initial guess
    X_init = InitialGuessList()
    q_ref = np.zeros(q_ref.shape)
    qdot_ref = np.zeros(qdot_ref.shape)
    X_init.add(np.concatenate([q_ref, qdot_ref]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialGuessList()
    tau_init = np.zeros(tau_init.shape)
    U_init.add(tau_init, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    parameters = ParameterList()
    bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
    # and an initial condition
    initial_gravity_orientation = InitialGuess([0, 0])
    parameters.add(
        parameter_name="gravity_angle",  # The name of the parameter
        function=rotating_gravity,  # The function that modifies the biorbd model
        bounds=bound_gravity,  # The bounds
        initial_guess=initial_gravity_orientation,  # The initial guess
        size=2,  # The number of elements this particular parameter vector has
    )

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        ode_solver=OdeSolver.RK8,
        nb_integration_steps=2,
        parameters=parameters,
        nb_threads=4,
        use_SX=use_ACADOS,
    )


if __name__ == "__main__":
    start = time.time()
    subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '44_1'
    print('Subject: ', subject, ', Trial: ', trial)

    use_ACADOS = False

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
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    q_ref_matlab = loadmat(kalman_path + q_name)['Q2']
    qdot_ref_matlab = loadmat(kalman_path + qd_name)['V2']
    qddot_ref_matlab = loadmat(kalman_path + qdd_name)['A2']

    load_path = 'Solutions/'
    load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)
    q_ref_biorbd = kalman_states['q']
    qdot_ref_biorbd = kalman_states['qd']
    qddot_ref_biorbd = kalman_states['qdd']

    initial_gravity = biorbd.Vector3d(0, 0, -9.80639)
    biorbd_model.setGravity(initial_gravity)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Calculate Kalman controls --- #
    q_ref_matlab = q_ref_matlab[:, frames.start:frames.stop:step_size]
    qdot_ref_matlab = qdot_ref_matlab[:, frames.start:frames.stop:step_size]
    qddot_ref_matlab = qddot_ref_matlab[:, frames.start:frames.stop:step_size]

    q_ref_biorbd = q_ref_biorbd[:, :frames.stop-frames.start:step_size]
    qdot_ref_biorbd = qdot_ref_biorbd[:, :frames.stop-frames.start:step_size]
    qddot_ref_biorbd = qddot_ref_biorbd[:, :frames.stop-frames.start:step_size]

    q_ref_matlab = correct_Kalman(biorbd_model, q_ref_matlab)
    q_ref_biorbd = correct_Kalman(biorbd_model, q_ref_biorbd)

    if subject == 'DoCi' or subject == 'BeLa' or subject == 'GuSe':
        q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab, q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd)
        q_ref[:6, :] = q_ref_biorbd[:6, :]
        qdot_ref[:6, :] = qdot_ref_biorbd[:6, :]
        qddot_ref[:6, :] = qddot_ref_biorbd[:6, :]
    if subject == 'JeCh':
        q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab)
    if subject == 'SaMi':
        if (trial == '821_seul_2' or trial == '821_seul_3'
            or trial == '821_contact_1' or trial == '821_contact_2' or trial == '821_contact_3'
            or trial == '821_822_4' or trial == '821_822_5'):
            q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab)
        else:
            q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab, q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd)

    states_idx_range_list, broken_dofs = check_Kalman(q_ref)
    if broken_dofs is not None:
        print('Abnormal Kalman states at DoFs: ', broken_dofs)
        for dof in broken_dofs:
            q_ref[dof, :] = 0
            qdot_ref[dof, :] = 0
            qddot_ref[dof, :] = 0

    tau_ref = inverse_dynamics(biorbd_model, q_ref, qdot_ref, qddot_ref)

    xmin, xmax = x_bounds(biorbd_model)

    markers_reordered, markers_idx_ref = reorder_markers(biorbd_model, c3d, frames, step_size, broken_dofs)
    markers_rotated = markers_reordered

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        q_ref=q_ref, qdot_ref=qdot_ref, tau_init=tau_ref, markers_ref=markers_rotated, markers_idx_ref=markers_idx_ref,
        states_idx_ref=states_idx_range_list, broken_dofs=broken_dofs,
        xmin=xmin, xmax=xmax,
        min_g=[0, -np.pi/16], max_g=[2*np.pi, np.pi/16],
        use_ACADOS=use_ACADOS,
    )

    # --- Solve the program --- #
    if use_ACADOS:
        options = {
            "integrator_type": "IRK",
            "nlp_solver_max_iter": 1000,
            "nlp_solver_step_length": 0.3,
            "nlp_solver_tol_comp": 1e-05,
            "nlp_solver_tol_eq": 1e-04,
            "nlp_solver_tol_ineq": 1e-06,
            "nlp_solver_tol_stat": 1e-06,
            "sim_method_newton_iter": 5,
            "sim_method_num_stages": 4,
            "sim_method_num_steps": 4}
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=options, show_online_optim=False)
    else:
        options = {"max_iter": 500, "tol": 1e-6, "constr_viol_tol": 1e-3, "linear_solver": "ma57"}
        sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)

    # --- Save --- #
    save_path = 'Solutions/'
    if use_ACADOS:
        save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_ACADOS'
    else:
        save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF_RK8'
    ocp.save(sol, save_name + ".bo")

    biorbd_model = biorbd.Model(model_path + model_name)
    biorbd_model.setGravity(initial_gravity)
    rotating_gravity(biorbd_model, params["gravity_angle"].squeeze())
    print_gravity = Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    gravity = print_gravity()['gravity'].full().squeeze()

    save_variables_name = save_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'states': states, 'controls': controls, 'params': params, 'gravity': gravity},
                    handle, protocol=3)

    angle = params["gravity_angle"].squeeze()/np.pi*180
    print('Number of shooting points: ', adjusted_number_shooting_points)
    print('Gravity rotation: ', angle)
    print('Gravity: ', gravity)

    if broken_dofs is not None:
        print('Abnormal Kalman states at DoFs: ', broken_dofs)

    stop = time.time()
    print('Runtime: ', stop - start)

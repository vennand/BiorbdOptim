import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import pickle
import os
import sys
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
from load_data_filename import load_data_filename
from x_bounds import x_bounds
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    InitialGuessList,
    InitialGuess,
    ShowResult,
    InterpolationType,
    Data,
    Solver,
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


# def refential_matrix(subject):
#     if subject == 'DoCi':
#         angleX = 0.0480
#         angleY = -0.0657
#         angleZ = 1.5720
#     elif subject == 'JeCh':
#         angleX = 0.0102;
#         angleY = 0.1869;
#         angleZ = -1.6201;
#
#     RotX = np.array(((1, 0, 0), (0, np.cos(angleX), -np.sin(angleX)), (0, np.sin(angleX), np.cos(angleX))))
#
#     RotY = np.array(((np.cos(angleY), 0, np.sin(angleY)), (0, 1, 0),(-np.sin(angleY), 0, np.cos(angleY))))
#
#     RotZ = np.array(((np.cos(angleZ), -np.sin(angleZ), 0), (np.sin(angleZ), np.cos(angleZ), 0), (0, 0, 1)))
#
#     return RotX.dot(RotY.dot(RotZ))


def prepare_ocp(biorbd_model, final_time, number_shooting_points, markers_ref, q_init, qdot_init, tau_init, xmin, xmax, use_ACADOS):
    # --- Options --- #
    torque_min, torque_max = -300, 300
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_init, qdot_init))
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1, target=markers_ref)
    # objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1e-7, target=tau_init)
    # objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1e-7)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1e-5, target=state_ref)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1e-5, states_idx=range(6, n_q))
    control_weight_segments = [0   , 0   , 0   ,  # pelvis trans
                               0   , 0   , 0   ,  # pelvis rot
                               1e-7, 1e-7, 1e-6,  # thorax
                               1e-5, 1e-5, 1e-4,  # head
                               1e-5, 1e-4,        # right shoulder
                               1e-5, 1e-5, 1e-4,  # right arm
                               1e-4, 1e-3,        # right forearm
                               1e-4, 1e-3,        # right hand
                               1e-5, 1e-4,        # left shoulder
                               1e-5, 1e-5, 1e-4,  # left arm
                               1e-4, 1e-3,        # left forearm
                               1e-4, 1e-3,        # left hand
                               1e-7, 1e-7, 1e-6,  # right thigh
                               1e-6,              # right leg
                               1e-4, 1e-3,        # right foot
                               1e-7, 1e-7, 1e-6,  # left thigh
                               1e-6,              # left leg
                               1e-4, 1e-3,        # left foot
                               ]
    for idx in range(n_tau):
      objective_functions.add(Objective.Lagrange.TRACK_TORQUE, weight=control_weight_segments[idx], target=tau_init, controls_idx=idx)
      objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=control_weight_segments[idx], controls_idx=idx)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    # constraints = ConstraintList()

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(Bounds(min_bound=xmin, max_bound=xmax))

    # Initial guess
    X_init = InitialGuessList()
    # q_init = np.zeros(q_init.shape)
    # qdot_init = np.zeros(qdot_init.shape)
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialGuessList()
    # tau_init = np.zeros(tau_init.shape)
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
        objective_functions,
        # constraints,
        nb_integration_steps=4,
        nb_threads=4,
        use_SX=use_ACADOS,
    )


if __name__ == "__main__":
    start = time.time()
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_seul_4'
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

    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # optimal_gravity_filename = '../optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".bo"
    # ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    # states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)

    filename = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".pkl"
    filename_full = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(frames.stop - frames.start - 1) + '_mixed_EKF' + ".pkl"
    filename_acados = '/home/andre/bioptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(adjusted_number_shooting_points) + '_mixed_EKF_ACADOS' + ".pkl"
    filename_acados_full = '/home/andre/bioptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(frames.stop - frames.start - 1) + '_mixed_EKF_ACADOS' + ".pkl"

    if os.path.isfile(filename_acados) and use_ACADOS:
        optimal_gravity_filename = filename_acados
    else:
        optimal_gravity_filename = filename
    if os.path.isfile(filename_acados_full) and use_ACADOS:
        optimal_gravity_filename_full = filename_acados_full
    else:
        optimal_gravity_filename_full = filename_full

    with open(optimal_gravity_filename, 'rb') as handle:
        data = pickle.load(handle)
    states_optimal_gravity = data['states']
    controls_optimal_gravity = data['controls']
    params_optimal_gravity_part = data['params']

    with open(optimal_gravity_filename_full, 'rb') as handle:
        data = pickle.load(handle)
    states_optimal_gravity_full = data['states']
    controls_optimal_gravity_full = data['controls']
    params_optimal_gravity_full = data['params']

    if subject == 'JeCh' and trial == '833_5':
        params_optimal_gravity = params_optimal_gravity_part
    elif subject == 'SaMi' and trial == '821_contact_2':
        states_optimal_gravity['q'] = states_optimal_gravity_full['q'][:, ::step_size]
        states_optimal_gravity['q_dot'] = states_optimal_gravity_full['q_dot'][:, ::step_size]
        controls_optimal_gravity['tau'] = controls_optimal_gravity_full['tau'][:, ::step_size]
        params_optimal_gravity = params_optimal_gravity_full
    else:
        params_optimal_gravity = params_optimal_gravity_full

    angle = params_optimal_gravity["gravity_angle"].squeeze()
    q_ref = states_optimal_gravity['q']
    qdot_ref = states_optimal_gravity['q_dot']
    tau_ref = controls_optimal_gravity['tau'][:, :-1]

    rotating_gravity(biorbd_model, angle)
    # print_gravity = Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    # print(print_gravity()['gravity'].full())

    xmin, xmax = x_bounds(biorbd_model)

    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    # markers_rotated = np.zeros(markers.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        markers_ref=markers_rotated, q_init=q_ref, qdot_init=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax,
        use_ACADOS=use_ACADOS,
    )

    # --- Solve the program --- #
    if use_ACADOS:
        options = {
            "integrator_type": "IRK",
            "nlp_solver_max_iter": 1000,
            "nlp_solver_step_length": 0.005,
            "nlp_solver_tol_comp": 1e-05,
            "nlp_solver_tol_eq": 1e-04,
            "nlp_solver_tol_ineq": 1e-06,
            "nlp_solver_tol_stat": 1e-06,
            "sim_method_newton_iter": 5,
            "sim_method_num_stages": 4,
            "sim_method_num_steps": 4}
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=options, show_online_optim=False)
    else:
        options = {"max_iter": 3000, "tol": 1e-4, "constr_viol_tol": 1e-2, "linear_solver": "ma57"}
        sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)

    # --- Save --- #
    save_path = 'Solutions/'
    if use_ACADOS:
        save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_ACADOS'
    else:
        save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
    ocp.save(sol, save_name + ".bo")

    get_gravity = Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    gravity = get_gravity()['gravity'].full().squeeze()

    save_variables_name = save_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'mocap': markers_rotated, 'duration': duration, 'frames': frames, 'step_size': step_size,
                     'states': states, 'controls': controls, 'gravity': gravity, 'gravity_angle': angle},
                    handle, protocol=3)

    stop = time.time()
    print('Number of shooting points: ', adjusted_number_shooting_points)
    print(stop - start)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
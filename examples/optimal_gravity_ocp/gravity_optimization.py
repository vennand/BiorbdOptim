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
from adjust_Kalman import adjust_Kalman

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    InitialConditionsList,
    InitialConditions,
    ShowResult,
    InterpolationType,
    Data,
    ParameterList,
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



def prepare_ocp(biorbd_model, final_time, number_shooting_points, q_ref, qdot_ref, tau_init, xmin, xmax, min_g, max_g, markers_ref=None, markers_idx_ref=None, states_idx_ref=None, min_torque_diff=False, broken_dofs=None):
    # --- Options --- #
    torque_min, torque_max = -300, 300
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_ref, qdot_ref))
    objective_functions = ObjectiveList()
    # if markers_ref is not None:
    #     objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=10, target=markers_ref, markers_idx=range(36, 39))
    #     objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=10, target=markers_ref, markers_idx=range(58, 61))
    #     objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=10, target=markers_ref, markers_idx=range(75, 78))
    #     objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=10, target=markers_ref, markers_idx=range(92, 95))
    if markers_idx_ref and markers_ref is not None:
        for markers_idx_range in markers_idx_ref:
            objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1, target=markers_ref,
                                    markers_idx=markers_idx_range)
    if states_idx_ref is not None:
        for states_idx_range in states_idx_ref:
            objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=1, target=state_ref,
                                    states_idx=states_idx_range)
    else:
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=1, target=state_ref,
             states_idx=range(n_q))
    # objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=0.01, target=state_ref,
    #      states_idx=range(n_q, n_q + n_qdot))
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1e-7)
    if min_torque_diff:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1e-5)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    # constraints = ConstraintList()

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(Bounds(min_bound=xmin, max_bound=xmax))
    # if markers_idx_ref is None and broken_dofs is not None:
    #     for dof in broken_dofs:
    #         X_bounds[0].min[dof, :] = 0
    #         X_bounds[0].max[dof, :] = 0
    # X_bounds[0].min[40:42, :] = 0
    # X_bounds[0].max[40:42, :] = 0


    # Initial guess
    X_init = InitialConditionsList()
    q_ref = np.zeros(q_ref.shape)
    qdot_ref = np.zeros(qdot_ref.shape)
    X_init.add(np.concatenate([q_ref, qdot_ref]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialConditionsList()
    tau_init = np.zeros(tau_init.shape)
    U_init.add(tau_init, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    parameters = ParameterList()
    bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
    # and an initial condition
    initial_gravity_orientation = InitialConditions([0, 0])
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
        # constraints,
        nb_integration_steps=4,
        parameters=parameters,
        nb_threads=4,
    )


if __name__ == "__main__":
    start = time.time()
    # subject = 'DoCi'
    subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 1000
    trial = '833_5'
    print('Subject: ', subject, ', Trial: ', trial)

    trial_needing_min_torque_diff = {'DoCi': ['44_1'],
                                     'BeLa': ['44_2'],
                                     'JeCh': ['833_5'],
                                     'SaMi': ['821_822_2',
                                              '821_contact_2',
                                              '821_seul_3', '821_seul_4']}
    min_torque_diff = False
    if subject in trial_needing_min_torque_diff.keys():
        if trial in trial_needing_min_torque_diff[subject]:
            min_torque_diff = True

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

    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
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

    q_ref_biorbd = q_ref_biorbd[:, ::step_size]
    qdot_ref_biorbd = qdot_ref_biorbd[:, ::step_size]
    qddot_ref_biorbd = qddot_ref_biorbd[:, ::step_size]

    q_ref, qdot_ref, qddot_ref = adjust_Kalman(biorbd_model, subject, trial,
                                               q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab,
                                               q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd)

    tau_ref = inverse_dynamics(biorbd_model, q_ref, qdot_ref, qddot_ref)

    # load_name = '/home/andre/BiorbdOptim/examples/optimal_estimation_no_constraint_ocp/Solutions/DoCi/Do_822_contact_2.bo'
    # load_name = '/home/andre/BiorbdOptim/examples/optimal_estimation_no_constraint_ocp/Solutions/DoCi/Do_822_contact_2_not_rotated.bo'
    # ocp_not_kalman, sol_not_kalman = OptimalControlProgram.load(load_name)
    # states_not_kalman, controls_not_kalman = Data.get_data(ocp_not_kalman, sol_not_kalman)
    # q_ref = states_not_kalman['q'][:, frames.start:frames.stop:step_size]
    # qdot_ref = states_not_kalman['q_dot'][:, frames.start:frames.stop:step_size]
    # tau_ref = controls_not_kalman['tau'][:, frames.start:frames.stop:step_size][:, :-1]

    xmin, xmax = x_bounds(biorbd_model)

    markers_reordered, markers_idx_ref = reorder_markers(biorbd_model, c3d, frames, step_size, broken_dofs)

    # markers_rotated = np.zeros(markers_reordered.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        q_ref=q_ref, qdot_ref=qdot_ref, tau_init=tau_ref, markers_ref=markers_rotated, markers_idx_ref=markers_idx_ref,
        states_idx_ref=states_idx_range_list, broken_dofs=broken_dofs, min_torque_diff=min_torque_diff,
        xmin=xmin, xmax=xmax,
        min_g=[0, -np.pi/16], max_g=[2*np.pi, np.pi/16],
    )

    # --- Solve the program --- #
    options = {"max_iter": 3000, "tol": 1e-6, "constr_viol_tol": 1e-3, "linear_solver": "ma57"}
    sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)

    # --- Save --- #
    save_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".bo"
    save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
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

    # --- Load --- #
    # ocp, sol = OptimalControlProgram.load(save_name)

    angle = params["gravity_angle"].squeeze()/np.pi*180
    print('Number of shooting points: ', adjusted_number_shooting_points)
    print('Gravity rotation: ', angle)
    print('Gravity: ', gravity)

    if broken_dofs is not None:
        print('Abnormal Kalman states at DoFs: ', broken_dofs)

    stop = time.time()
    print('Runtime: ', stop - start)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)

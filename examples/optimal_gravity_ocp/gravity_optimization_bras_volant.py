import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import os, sys
import warnings
from load_data_filename import load_data_filename
from x_bounds import x_bounds

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


def inverse_dynamics(biorbd_model, q_ref, qd_ref, qdd_ref):
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    return id(q_ref, qd_ref, qdd_ref)[:, :-1]


# def check_Kalman(q_ref):
#     segments_with_pi_limits = [15, 17, 24, 26, 18, 27, 31, 37, 34, 40]
#     bool = np.zeros(q_ref[6:, :].shape)
#     for (i, j), q in np.ndenumerate(q_ref[6:, :]):
#         if i+6+1 in segments_with_pi_limits:
#             bool[i, j] = ((q / np.pi).astype(int) != 0)
#         else:
#             bool[i, j] = ((q / (np.pi/2)).astype(int) != 0)
#     # bool = ((q_ref[6:, :] / (2*np.pi)).astype(int) != 0)
#     return bool.any()


def check_Kalman(q_ref):
    segments_with_pi_limits = [15, 17, 18, 24, 26, 27]#, 31, 34, 37, 40]
    bool = np.zeros(q_ref[:30, :].shape)
    for (i, j), q in np.ndenumerate(q_ref[6:30, :]):
        if i+6+1 in segments_with_pi_limits:
            bool[i+6, j] = ((q / np.pi).astype(int) != 0)
        else:
            bool[i+6, j] = ((q / (np.pi/2)).astype(int) != 0)
    states_idx_bool = bool.any(axis=1)

    states_idx_range_list = []
    start_index = 6
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


# def correct_Kalman(q_ref, qdot_ref, qddot_ref, frequency):
#     # segments_with_pi_limits = [15, 17, 24, 26, 18, 27, 31, 37, 34, 40]
#     # for (i, j), q in np.ndenumerate(q_ref[6:, :]):
#     #     if i + 6 + 1 in segments_with_pi_limits:
#     #         q_ref[i+6, j] = q - ((q / np.pi).astype(int) * np.pi)
#     #     else:
#     #         q_ref[i+6, j] = q - ((q / (np.pi / 2)).astype(int) * (np.pi / 2))
#     q_ref[6:, :] = q_ref[6:, :] - ((q_ref[6:, :] / (2*np.pi)).astype(int) * (2*np.pi))
#     # qdot_ref[6:, :-1] = (q_ref[6:, 1:] - q_ref[6:, :-1]) * frequency
#     # qddot_ref[6:, :-1] = (qdot_ref[6:, 1:] - qdot_ref[6:, :-1]) * frequency
#
#     return q_ref, qdot_ref, qddot_ref


def prepare_ocp(biorbd_model, final_time, number_shooting_points, q_ref, qdot_ref, tau_init, xmin, xmax, min_g, max_g, markers_ref=None, markers_idx_ref=None, states_idx_ref=None):
    # --- Options --- #
    torque_min, torque_max = -150, 150
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

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    # constraints = ConstraintList()

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(Bounds(min_bound=xmin, max_bound=xmax))
    # for idx_q_fix, q_fix in enumerate(q_ref[:6, 0]):
    #     X_bounds[0].min[idx_q_fix, :] = q_fix
    #     X_bounds[0].max[idx_q_fix, :] = q_fix
    X_bounds[0].min[:6, :] = 0
    X_bounds[0].max[:6, :] = 0
    X_bounds[0].min[30:n_q, :] = 0
    X_bounds[0].max[30:n_q, :] = 0
    X_bounds[0].min[n_q:n_q+6, :] = 0
    X_bounds[0].max[n_q:n_q+6, :] = 0
    X_bounds[0].min[n_q+30:n_q+n_q, :] = 0
    X_bounds[0].max[n_q+30:n_q+n_q, :] = 0


    # Initial guess
    X_init = InitialConditionsList()
    X_init.add(np.concatenate([q_ref, qdot_ref]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    # U_bounds[0].min[:6, :] = 0
    # U_bounds[0].max[:6, :] = 0

    U_init = InitialConditionsList()
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
        # parameters=parameters,
        nb_threads=4,
    )


if __name__ == "__main__":
    start = time.time()
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 5000
    trial = 'bras_volant_1'

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
    q_ref = loadmat(kalman_path + q_name)['Q2']
    qdot_ref = loadmat(kalman_path + qd_name)['V2']
    qddot_ref = loadmat(kalman_path + qdd_name)['A2']

    # biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, 0))

    # --- Adjust number of shooting points --- #
    list_adjusted_number_shooting_points = []
    for frame_num in range(1, (frames.stop - frames.start - 1) // frames.step + 1):
        list_adjusted_number_shooting_points.append((frames.stop - frames.start - 1) // frame_num + 1)
    diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
    step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
    adjusted_number_shooting_points = ((frames.stop - frames.start - 1) // step_size + 1) - 1

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Calculate Kalman controls --- #
    q_ref = q_ref[:, frames.start:frames.stop:step_size]
    qdot_ref = qdot_ref[:, frames.start:frames.stop:step_size]
    qddot_ref = qddot_ref[:, frames.start:frames.stop:step_size]
    tau_ref = inverse_dynamics(biorbd_model, q_ref, qdot_ref, qddot_ref)

    # if check_Kalman(q_ref):
    #     warnings.warn('Corrected abnormal Kalman states')
    #     q_ref, qdot_ref, qddot_ref = correct_Kalman(q_ref, qdot_ref, qddot_ref, frequency)
    states_idx_range_list, broken_dofs = check_Kalman(q_ref)
    if broken_dofs is not None:
        print('Abnormal Kalman states at DoFs: ', broken_dofs)

    # load_name = '/home/andre/BiorbdOptim/examples/optimal_estimation_no_constraint_ocp/Solutions/DoCi/Do_822_contact_2.bo'
    # load_name = '/home/andre/BiorbdOptim/examples/optimal_estimation_no_constraint_ocp/Solutions/DoCi/Do_822_contact_2_not_rotated.bo'
    # ocp_not_kalman, sol_not_kalman = OptimalControlProgram.load(load_name)
    # states_not_kalman, controls_not_kalman = Data.get_data(ocp_not_kalman, sol_not_kalman)
    # q_ref = states_not_kalman['q'][:, frames.start:frames.stop:step_size]
    # qdot_ref = states_not_kalman['q_dot'][:, frames.start:frames.stop:step_size]
    # tau_ref = controls_not_kalman['tau'][:, frames.start:frames.stop:step_size][:, :-1]

    xmin, xmax = x_bounds(biorbd_model)

    markers = c3d['data']['points'][:3, :95, frames.start:frames.stop:step_size] / 1000
    c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
    model_labels = [label.to_string() for label in biorbd_model.markerNames()]
    labels_index = [index_c3d for label in model_labels for index_c3d, c3d_label in enumerate(c3d_labels) if label in c3d_label]
    markers_reordered = np.zeros((3, len(labels_index), markers.shape[2]))
    for index, label_index in enumerate(labels_index):
        markers_reordered[:, index, :] = markers[:, label_index, :]

    # markers_rotated = np.zeros(markers_reordered.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    model_segments = {
        'pelvis': {'markers': ['EIASD', 'CID', 'EIPSD', 'EIPSG', 'CIG', 'EIASG'], 'dofs': range(0, 6)},
        'thorax': {'markers': ['MANU', 'MIDSTERNUM', 'XIPHOIDE', 'C7', 'D3', 'D10'], 'dofs': range(6, 9)},
        'head': {'markers': ['ZYGD', 'TEMPD', 'GLABELLE', 'TEMPG', 'ZYGG'], 'dofs': range(9, 12)},
        'right_shoulder': {'markers': ['CLAV1D', 'CLAV2D', 'CLAV3D', 'ACRANTD', 'ACRPOSTD', 'SCAPD'], 'dofs': range(12, 14)},
        'right_arm': {'markers': ['DELTD', 'BICEPSD', 'TRICEPSD', 'EPICOND', 'EPITROD'], 'dofs': range(14, 17)},
        'right_forearm': {'markers': ['OLE1D', 'OLE2D', 'BRACHD', 'BRACHANTD', 'ABRAPOSTD', 'ABRASANTD', 'ULNAD', 'RADIUSD'], 'dofs': range(17, 19)},
        'right_hand': {'markers': ['METAC5D', 'METAC2D', 'MIDMETAC3D'], 'dofs': range(19, 21)},
        'left_shoulder': {'markers': ['CLAV1G', 'CLAV2G', 'CLAV3G', 'ACRANTG', 'ACRPOSTG', 'SCAPG'], 'dofs': range(21, 23)},
        'left_arm': {'markers': ['DELTG', 'BICEPSG', 'TRICEPSG', 'EPICONG', 'EPITROG'], 'dofs': range(23, 26)},
        'left_forearm': {'markers': ['OLE1G', 'OLE2G', 'BRACHG', 'BRACHANTG', 'ABRAPOSTG', 'ABRANTG', 'ULNAG', 'RADIUSG'], 'dofs': range(26, 28)},
        'left_hand': {'markers': ['METAC5G', 'METAC2G', 'MIDMETAC3G'], 'dofs': range(28, 30)},
        'right_thigh': {'markers': ['ISCHIO1D', 'TFLD', 'ISCHIO2D', 'CONDEXTD', 'CONDINTD'], 'dofs': range(30, 33)},
        'right_leg': {'markers': ['CRETED', 'JAMBLATD', 'TUBD', 'ACHILED', 'MALEXTD', 'MALINTD'], 'dofs': range(33, 34)},
        'right_foot': {'markers': ['CALCD', 'MIDMETA4D', 'MIDMETA1D', 'SCAPHOIDED', 'METAT5D', 'METAT1D'], 'dofs': range(34, 36)},
        'left_thigh': {'markers': ['ISCHIO1G', 'TFLG', 'ISCHIO2G', 'CONEXTG', 'CONDINTG'], 'dofs': range(36, 39)},
        'left_leg': {'markers': ['CRETEG', 'JAMBLATG', 'TUBG', 'ACHILLEG', 'MALEXTG', 'MALINTG', 'CALCG'], 'dofs': range(39, 40)},
        'left_foot': {'markers': ['MIDMETA4G', 'MIDMETA1G', 'SCAPHOIDEG', 'METAT5G', 'METAT1G'], 'dofs': range(40, 42)},
    }

    markers_idx_ref = []
    if broken_dofs is not None:
        for dof in broken_dofs:
            for segment in model_segments.values():
                if dof in segment['dofs']:
                    marker_positions = [index_model for marker_label in segment['markers'] for index_model, model_label in enumerate(model_labels) if marker_label in model_label]
                    if range(min(marker_positions), max(marker_positions) + 1) not in markers_idx_ref:
                        markers_idx_ref.append(range(min(marker_positions), max(marker_positions) + 1))

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        q_ref=q_ref, qdot_ref=qdot_ref, tau_init=tau_ref, markers_ref=markers_rotated, markers_idx_ref=markers_idx_ref,
        states_idx_ref=states_idx_range_list,
        xmin=xmin, xmax=xmax,
        min_g=[0, -np.pi/16], max_g=[2*np.pi, np.pi/16],
    )

    # --- Solve the program --- #
    # options = {"max_iter": 3000, "tol": 1e-6, "constr_viol_tol": 1e-3, "linear_solver": "ma57"}
    # sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)
    sol = ocp.solve(show_online_optim=False)

    # --- Save --- #
    save_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_onlyQ' + ".bo"
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + ".bo"
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_one_core' + ".bo"
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_EndChainMarkers' + ".bo"
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_not_kalman_EndChainMarkers' + ".bo"
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_not_kalman_not_rotated_EndChainMarkers' + ".bo"
    save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_rotated_model' + ".bo"
    ocp.save(sol, save_name)

    # --- Load --- #
    # ocp, sol = OptimalControlProgram.load(save_name)

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)
    print('Number of shooting points: ', adjusted_number_shooting_points)

    if broken_dofs is not None:
        print('Abnormal Kalman states at DoFs: ', broken_dofs)

    stop = time.time()
    print('Runtime: ', stop - start)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)

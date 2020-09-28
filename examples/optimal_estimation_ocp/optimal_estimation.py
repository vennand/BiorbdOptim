import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX
import pickle
import os
import sys
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
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


def prepare_ocp(biorbd_model, final_time, number_shooting_points, markers_ref, q_init, qdot_init, tau_init, xmin, xmax):
    # --- Options --- #
    torque_min, torque_max = -150, 150
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_ref, qdot_ref))
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1, target=markers_ref)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1e-7)
    # control_weight_segments = [1e-7, 1e-7, 1e-7,  # pelvis trans
    #                            1e-7, 1e-7, 1e-7,  # pelvis rot
    #                            1e-7, 1e-7, 1e-7,  # thorax
    #                            1e-5, 1e-5, 1e-5,  # head
    #                            1e-5, 1e-5,        # right shoulder
    #                            1e-5, 1e-5, 1e-5,  # right arm
    #                            1e-4, 1e-4,        # right forearm
    #                            1e-4, 1e-4,        # right hand
    #                            1e-5, 1e-5,        # left shoulder
    #                            1e-5, 1e-5, 1e-5,  # left arm
    #                            1e-4, 1e-4,        # left forearm
    #                            1e-4, 1e-4,        # left hand
    #                            1e-7, 1e-7, 1e-7,  # right thigh
    #                            1e-6,              # right leg
    #                            1e-4, 1e-4,        # right foot
    #                            1e-7, 1e-7, 1e-7,  # left thigh
    #                            1e-6,              # left leg
    #                            1e-4, 1e-4,        # left foot
    #                            ]
    # for idx in range(n_tau):
    #   objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=control_weight_segments[idx], controls_idx=idx)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    # constraints = ConstraintList()

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(Bounds(min_bound=xmin, max_bound=xmax))

    # Initial guess
    X_init = InitialConditionsList()
    # q_init = np.zeros(q_init.shape)
    # qdot_init = np.zeros(qdot_init.shape)
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialConditionsList()
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
    )


if __name__ == "__main__":
    start = time.time()
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_seul_1'
    use_OGE = True

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

    # Adjust number of shooting points
    list_adjusted_number_shooting_points = []
    for frame_num in range(1, (frames.stop - frames.start - 1) // frames.step + 1):
        list_adjusted_number_shooting_points.append((frames.stop - frames.start - 1) // frame_num + 1)
    diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
    step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
    adjusted_number_shooting_points = ((frames.stop - frames.start - 1) // step_size + 1) - 1

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    if use_OGE:
        optimal_gravity_filename = '../optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".bo"
        ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
        states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)

        angle = params_optimal_gravity["gravity_angle"].squeeze()
        q_ref = states_optimal_gravity['q']
        qdot_ref = states_optimal_gravity['q_dot']
        tau_ref = controls_optimal_gravity['tau'][:, :-1]

        rotating_gravity(biorbd_model, angle)
    else:
        q_ref = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size]
        qdot_ref = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
        qddot_ref = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]
        tau_ref = inverse_dynamics(biorbd_model, q_ref, qdot_ref, qddot_ref)

        # load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
        # load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
        # with open(load_variables_name, 'rb') as handle:
        #     kalman_states = pickle.load(handle)
        # q_ref = kalman_states['q']
        # qdot_ref = kalman_states['qd']
        # qddot_ref = kalman_states['qdd']


    xmin, xmax = x_bounds(biorbd_model)

    markers = c3d['data']['points'][:3, :95, frames.start:frames.stop:step_size] / 1000
    c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
    model_labels = [label.to_string() for label in biorbd_model.markerNames()]
    # labels_index = [index_c3d for label in model_labels for index_c3d, c3d_label in enumerate(c3d_labels) if label in c3d_label]

    ### --- Test --- ###
    labels_index = []
    missing_markers_index = []
    for index_model, model_label in enumerate(model_labels):
        missing_markers_bool = True
        for index_c3d, c3d_label in enumerate(c3d_labels):
            if model_label in c3d_label:
                labels_index.append(index_c3d)
                missing_markers_bool = False
        if missing_markers_bool:
            labels_index.append(index_model)
            missing_markers_index.append(index_model)
    ### --- Test --- ###

    # markers_reordered = np.zeros((3, len(labels_index), markers.shape[2]))
    # for index, label_index in enumerate(labels_index):
    #     markers_reordered[:, index, :] = markers[:, label_index, :]
    markers_reordered = np.zeros((3, markers.shape[1], markers.shape[2]))
    for index, label_index in enumerate(labels_index):
        if index in missing_markers_index:
            markers_reordered[:, index, :] = np.nan
        else:
            markers_reordered[:, index, :] = markers[:, label_index, :]

    # markers_rotated = np.zeros(markers.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        markers_ref=markers_rotated, q_init=q_ref, qdot_init=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax,
    )

    # --- Solve the program --- #
    options = {"max_iter": 3000, "tol": 1e-4, "constr_viol_tol": 1e-2, "linear_solver": "ma57"}
    sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Save --- #
    # save_name = "Do_822_contact_2" + "_optimal_estimation_N" + str(adjusted_number_shooting_points)# + "_variable_weights" # + "_U10-5"
    save_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
    if use_OGE:
        save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
    else:
        save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_bad_EKF'
    ocp.save(sol, save_name + ".bo")

    save_variables_name = save_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'mocap': markers_rotated, 'duration': duration, 'frames': frames, 'step_size': step_size},
                    handle, protocol=3)

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)

    stop = time.time()
    print('Number of shooting points: ', adjusted_number_shooting_points)
    print(stop - start)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX
import pickle
import os

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


def refential_matrix(subject):
    if subject == 'DoCi':
        angleX = 0.0480
        angleY = -0.0657
        angleZ = 1.5720
    elif subject == 'JeCh':
        angleX = 0.0102;
        angleY = 0.1869;
        angleZ = -1.6201;

    RotX = np.array(((1, 0, 0), (0, np.cos(angleX), -np.sin(angleX)), (0, np.sin(angleX), np.cos(angleX))))

    RotY = np.array(((np.cos(angleY), 0, np.sin(angleY)), (0, 1, 0),(-np.sin(angleY), 0, np.cos(angleY))))

    RotZ = np.array(((np.cos(angleZ), -np.sin(angleZ), 0), (np.sin(angleZ), np.cos(angleZ), 0), (0, 0, 1)))

    return RotX.dot(RotY.dot(RotZ))


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
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

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
        objective_functions,
        # constraints,
        nb_integration_steps=4,
        #nb_threads=4,
        continuity=False,
    )


def x_bounds(biorbd_model):
    pi = np.pi
    inf = 50000
    n_qdot = biorbd_model.nbQdot()

    qmin_base = [-inf, -inf, -inf, -inf, -pi / 2.1, -inf]
    qmax_base = [inf, inf, inf, inf, pi / 2.1, inf]
    qmin_thorax = [-pi / 2, -pi / 2.1, -pi / 2]
    qmax_thorax = [pi / 2, pi / 2.1, pi / 2]
    qmin_tete = [-pi / 2, -pi / 2.1, -pi / 2]
    qmax_tete = [pi / 2, pi / 2.1, pi / 2]
    qmin_epaule_droite = [-pi / 2, -pi / 2]
    qmax_epaule_droite = [pi / 2, pi / 2]
    qmin_bras_droit = [-pi, -pi / 2.1, -pi]
    qmax_bras_droit = [pi, pi / 2.1, pi]
    qmin_avantbras_droit = [0, -pi / 2]
    qmax_avantbras_droit = [pi, pi / 2]
    qmin_main_droite = [-pi / 2, -pi / 2]
    qmax_main_droite = [pi / 2, pi / 2]
    qmin_epaule_gauche = [-pi / 2, -pi / 2]
    qmax_epaule_gauche = [pi / 2, pi / 2]
    qmin_bras_gauche = [-pi, -pi / 2.1, -pi]
    qmax_bras_gauche = [pi, pi / 2.1, pi]
    qmin_avantbras_gauche = [0, -pi / 2]
    qmax_avantbras_gauche = [pi, pi / 2]
    qmin_main_gauche = [-pi / 2, -pi / 2]
    qmax_main_gauche = [pi / 2, pi / 2]
    qmin_cuisse_droite = [-pi, -pi / 2.1, -pi / 2]
    qmax_cuisse_droite = [pi, pi / 2.1, pi / 2]
    qmin_jambe_droite = [-pi]
    qmax_jambe_droite = [0]
    qmin_pied_droit = [-pi / 2, -pi / 2]
    qmax_pied_droit = [pi / 2, pi / 2]
    qmin_cuisse_gauche = [-pi, -pi / 2.1, -pi / 2]
    qmax_cuisse_gauche = [pi, pi / 2.1, pi / 2]
    qmin_jambe_gauche = [-pi]
    qmax_jambe_gauche = [0]
    qmin_pied_gauche = [-pi / 2, -pi / 2]
    qmax_pied_gauche = [pi / 2, pi / 2]

    qdotmin_base = [-inf, -inf, -inf, -inf, -inf, -inf]
    qdotmax_base = [inf, inf, inf, inf, inf, inf]

    xmin = (qmin_base +  # q
            qmin_thorax +
            qmin_tete +
            qmin_epaule_droite +
            qmin_bras_droit +
            qmin_avantbras_droit +
            qmin_main_droite +
            qmin_epaule_gauche +
            qmin_bras_gauche +
            qmin_avantbras_gauche +
            qmin_main_gauche +
            qmin_cuisse_droite +
            qmin_jambe_droite +
            qmin_pied_droit +
            qmin_cuisse_gauche +
            qmin_jambe_gauche +
            qmin_pied_gauche +
            qdotmin_base +  # qdot
            [-200] * (n_qdot - 6))

    xmax = (qmax_base +
            qmax_thorax +
            qmax_tete +
            qmax_epaule_droite +
            qmax_bras_droit +
            qmax_avantbras_droit +
            qmax_main_droite +
            qmax_epaule_gauche +
            qmax_bras_gauche +
            qmax_avantbras_gauche +
            qmax_main_gauche +
            qmax_cuisse_droite +
            qmax_jambe_droite +
            qmax_pied_droit +
            qmax_cuisse_gauche +
            qmax_jambe_gauche +
            qmax_pied_gauche +
            qdotmax_base +  # qdot
            [200] * (n_qdot - 6))

    return xmin, xmax


if __name__ == "__main__":
    start = time.time()
    subject = 'DoCi'
    # subject = 'JeCh'

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'
    kalman_path = data_path + 'Q/'
    if subject == 'DoCi':
        model_name = 'DoCi.s2mMod'
        c3d_name = 'Do_822_contact_2.c3d'
    elif subject == 'JeCh':
        model_name = 'JeCh_201.s2mMod'
        c3d_name = 'Je_833_1.c3d'
    elif subject == 'BeLa':
        raise Exception(subject + ' is not available yet')
    else:
        raise Exception(subject + ' is not a valid subject')

    biorbd_model = biorbd.Model(model_path + model_name)
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    markers = c3d['data']['points'][:3, :95, :] / 1000
    c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
    model_labels = [label.to_string() for label in biorbd_model.markerNames()]
    labels_index = [np.where(np.isin(c3d_labels, label))[0][0] for label in model_labels]
    markers_reordered = np.zeros(markers.shape)
    for index, label_index in enumerate(labels_index):
        markers_reordered[:, index, :] = markers[:, label_index, :]

    # markers_rotated = np.zeros(markers.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    frames = markers.shape[2]
    number_shooting_points = frames-1

    frequency = c3d['header']['points']['frame_rate']
    duration = frames / frequency

    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    q_ref = np.zeros((biorbd_model.nbQ(), frames))
    qdot_ref = np.zeros((biorbd_model.nbQdot(), frames))
    tau_ref = np.zeros((biorbd_model.nbQddot(), number_shooting_points))

    xmin, xmax = x_bounds(biorbd_model)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=number_shooting_points,
        markers_ref=markers_rotated, q_init=q_ref, qdot_init=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax,
    )

    # --- Solve the program --- #
    options = {"max_iter": 3000, "tol": 1e-6, "constr_viol_tol": 1e-3}
    sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Save --- #
    save_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_no_constraint_ocp/Solutions/'
    # save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + ".bo"
    save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + '_not_rotated' + ".bo"
    ocp.save(sol, save_name)

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)

    stop = time.time()
    print(stop - start)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=number_shooting_points)
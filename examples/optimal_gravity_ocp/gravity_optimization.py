import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Bounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
)


def my_parameter_function(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


def prepare_ocp(biorbd_model, final_time, number_shooting_points, q_ref, qdot_ref, tau_init, xmin, xmax, min_g, max_g):
    # --- Options --- #
    torque_min, torque_max = -100, 100
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.TRACK_STATE, "weight": 1, "data_to_track": q_ref.T, "states_idx": range(n_q)},
        {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "data_to_track": qdot_ref.T,
         "states_idx": range(n_qdot)},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1e-7}
    )

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = Bounds(min_bound=xmin, max_bound=xmax)

    # Initial guess
    X_init = InitialConditions(np.concatenate([q_ref, qdot_ref]), interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    U_bounds.min[:6, :] = 0
    U_bounds.max[:6, :] = 0

    U_init = InitialConditions(tau_init, interpolation_type=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation_type=InterpolationType.CONSTANT)
    # and an initial condition
    initial_gravity_orientation = InitialConditions([0, 0])
    parameters = {
        "name": "gravity_angle",  # The name of the parameter
        "function": my_parameter_function,  # The function that modifies the biorbd model
        "bounds": bound_gravity,  # The bounds
        "initial_guess": initial_gravity_orientation,  # The initial guess
        "size": 2,  # The number of elements this particular parameter vector has
    }

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        parameters=parameters,
    )


def x_bounds(biorbd_model):
    pi = np.pi
    inf = np.inf
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
    biorbd_model = biorbd.Model("DoCi.s2mMod")

    frames = range(3099, 3300)
    number_shooting_points = 50
    step_size = int((len(frames)-1)/number_shooting_points)
    c3d = ezc3d.c3d('Do_822_contact_2.c3d');
    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency;

    from casadi import MX
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    q_ref = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat')['Q2'][:, 3099:3300:step_size]
    qdot_ref = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat')['V2'][:, 3099:3300:step_size]
    qddot_ref = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat')['A2'][:, 3099:3300:step_size]

    tau_ref = id(q_ref, qdot_ref, qddot_ref)
    tau_ref = tau_ref[:, :-1]

    # n_q = biorbd_model.nbQ()
    # n_qdot = biorbd_model.nbQdot()
    # q_ref = [0] * n_q
    # qdot_ref = [0] * n_qdot

    xmin, xmax = x_bounds(biorbd_model)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=3, number_shooting_points=number_shooting_points,
        q_ref=q_ref, qdot_ref=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax,
        min_g=[0, -np.pi], max_g=[np.pi, np.pi],
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Save --- #
    save_name = "test.bo"
    ocp.save(sol, save_name)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    length = params["gravity_angle"][0, 0]
    print(length)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=50)

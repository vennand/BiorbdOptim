import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX

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


def prepare_ocp(biorbd_model, final_time, number_shooting_points, q_ref, qdot_ref, tau_init, xmin, xmax, min_g, max_g):
    # --- Options --- #
    torque_min, torque_max = -100, 100
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_ref, qdot_ref))
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=1, target=state_ref,
         states_idx=range(n_q))
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=0.01, target=state_ref,
         states_idx=range(n_q, n_q + n_qdot))
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1e-7)

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
    X_init.add(np.concatenate([q_ref, qdot_ref]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

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
        parameters=parameters,
        nb_threads=4,
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
    biorbd_model = biorbd.Model("DoCi.s2mMod")

    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    c3d = ezc3d.c3d('Do_822_contact_2.c3d')
    frames = range(3099, 3300)

    number_shooting_points = 50
    # Adjust number of shooting points
    list_adjusted_number_shooting_points = []
    for frame_num in range(1, (frames.stop - frames.start - 1) // frames.step + 1):
        list_adjusted_number_shooting_points.append((frames.stop - frames.start - 1) // frame_num + 1)
    diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
    step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
    adjusted_number_shooting_points = ((frames.stop - frames.start - 1) // step_size + 1) - 1

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    q_ref = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat')['Q2'][:, frames.start:frames.stop:step_size]
    qdot_ref = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat')['V2'][:, frames.start:frames.stop:step_size]
    qddot_ref = loadmat('Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat')['A2'][:, frames.start:frames.stop:step_size]

    tau_ref = id(q_ref, qdot_ref, qddot_ref)
    tau_ref = tau_ref[:, :-1]

    xmin, xmax = x_bounds(biorbd_model)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        q_ref=q_ref, qdot_ref=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax,
        min_g=[0, -np.pi/16], max_g=[2*np.pi, np.pi/16],
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Save --- #
    save_name = "Do_822_contact_2_optimal_gravity_N" + str(adjusted_number_shooting_points) + ".bo"
    ocp.save(sol, save_name)

    # --- Load --- #
    # ocp, sol = OptimalControlProgram.load(save_name)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    angle = params["gravity_angle"]/np.pi*180
    print(angle)

    stop = time.time()
    print(stop - start)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)

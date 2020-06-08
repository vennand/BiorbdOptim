import biorbd
import numpy as np
import pickle
from time import time

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Bounds,
    QAndQDotBounds,
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
    gravity.applyRT(biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, q_ref, min_g, max_g):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -30, 30, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.TRACK_STATE, "weight": 1, "data_to_track": q_ref, "states_idx": range(biorbd_model.nbQ())},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1e-7}
    )

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, [0, -1]] = 0
    X_bounds.max[:, [0, -1]] = 0
    X_bounds.min[1, -1] = 3.14
    X_bounds.max[1, -1] = 3.14

    # Initial guess
    X_init = InitialConditions([0] * (n_q + n_qdot))

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    U_bounds.min[1, :] = 0
    U_bounds.max[1, :] = 0

    U_init = InitialConditions([torque_init] * n_tau)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation_type=InterpolationType.CONSTANT)
    # and an initial condition
    initial_gravity = InitialConditions([0, 0])
    parameters = {
        "name": "gravity_angle",  # The name of the parameter
        "function": my_parameter_function,  # The function that modifies the biorbd model
        "bounds": bound_gravity,  # The bounds
        "initial_guess": initial_gravity,  # The initial guess
        "size": 2,  # The number of elements this particular parameter vector has
        "type": Objective.Mayer,  # The type objective or constraint function (if there is any)
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


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, min_g=[0, -np.pi], max_g=[np.pi, np.pi],
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    length = params["gravity_angle"][0, 0]
    print(length)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=200)

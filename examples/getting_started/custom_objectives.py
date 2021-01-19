"""
File that shows an example of a custom constraint.
As an example, this custom constraint reproduces exactly the behavior of the ALIGN_MARKERS constraint.
"""
import biorbd
from casadi import vertcat

from bioptim import (
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ObjectiveFcn,
    ObjectiveList,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    OdeSolver,
)


def custom_func_align_markers(ocp, nlp, t, x, u, p, first_marker_idx, second_marker_idx):
    nq = nlp.shape["q"]
    val = []
    markers_func = biorbd.to_casadi_func("markers", nlp.model.markers, nlp.q)
    for v in x:
        q = v[:nq]
        markers = markers_func(q)
        first_marker = markers[:, first_marker_idx]
        second_marker = markers[:, second_marker_idx]
        val = vertcat(val, first_marker - second_marker)
    return val


def prepare_ocp(biorbd_model_path, ode_solver=OdeSolver.RK4):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)
    objective_functions.add(
        custom_func_align_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.START,
        quadratic=True,
        first_marker_idx=0,
        second_marker_idx=1,
        weight=1000,
    )
    objective_functions.add(
        custom_func_align_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        quadratic=True,
        first_marker_idx=0,
        second_marker_idx=2,
        weight=1000,
    )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    # Initial guess
    x_init = InitialGuess([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = Bounds([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuess([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    model_path = "cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()

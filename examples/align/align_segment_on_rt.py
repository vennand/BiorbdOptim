import biorbd

from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, ode_solver=OdeSolver.RK4):
    # --- Options --- #nq
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.ALIGN_SEGMENT_WITH_CUSTOM_RT, node=Node.ALL, segment_idx=2, rt_idx=0)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][2, [0, -1]] = [-1.57, 1.57]
    x_bounds[0][nq:, [0, -1]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque())

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
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=1,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()

import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    ObjectiveList,
    Objective,
    ConstraintList,
    Constraint,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    ShowResult,
    OdeSolver,
    Data,
)


def prepare_ocp(
    final_time, time_min, time_max, number_shooting_points, biorbd_model_path="cube.bioMod", ode_solver=OdeSolver.RK
):
    # --- Options --- #
    nb_phases = len(number_shooting_points)
    if nb_phases != 1 and nb_phases != 3:
        raise RuntimeError("Number of phases must be 1 to 3")

    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=0)
    if nb_phases == 3:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=1)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=0)
    if nb_phases == 3:
        dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=1)
        dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=2)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=0)
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.END, first_marker_idx=0, second_marker_idx=2, phase=0)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0], phase=0)
    if nb_phases == 3:
        constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.END, first_marker_idx=0, second_marker_idx=1, phase=1)
        constraints.add(
            Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[1], maximum=time_max[1], phase=1
        )
        constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.END, first_marker_idx=0, second_marker_idx=2, phase=2)
        constraints.add(
            Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[2], maximum=time_max[2], phase=2
        )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model[0]))  # Phase 0
    if nb_phases == 3:
        x_bounds.add(QAndQDotBounds(biorbd_model[0]))  # Phase 1
        x_bounds.add(QAndQDotBounds(biorbd_model[0]))  # Phase 2

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds.min[i, [0, -1]] = 0
            bounds.max[i, [0, -1]] = 0
    x_bounds[0].min[2, 0] = 0.0
    x_bounds[0].max[2, 0] = 0.0
    if nb_phases == 3:
        x_bounds[2].min[2, [0, -1]] = [0.0, 1.57]
        x_bounds[2].max[2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    if nb_phases == 3:
        x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
        x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()])
    if nb_phases == 3:
        u_bounds.add(
            [[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()]
        )
        u_bounds.add(
            [[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()]
        )

    u_init = InitialConditionsList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    if nb_phases == 3:
        u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
        u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model[:nb_phases],
        dynamics,
        number_shooting_points,
        final_time[:nb_phases],
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    ns = (20, 30, 20)
    ocp = prepare_ocp(final_time=final_time, time_min=time_min, time_max=time_max, number_shooting_points=ns)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    print(f"The optimized phase time are: {param['time'][0, 0]}s, {param['time'][1, 0]}s and {param['time'][2, 0]}s.")

    result = ShowResult(ocp, sol)
    result.animate()

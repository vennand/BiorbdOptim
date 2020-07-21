import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    ShowResult,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -1, 1, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE)
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL)
    objective_functions.add(Objective.Mayer.ALIGN_MARKERS, first_marker_idx=0, second_marker_idx=5)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:, 0] = (0, 0.07, 1.4, 0, 0, 0)
    x_bounds[0].max[:, 0] = (0, 0.07, 1.4, 0, 0, 0)

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    u_init = InitialConditionsList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
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
    )


if __name__ == "__main__":
    ocp = prepare_ocp(biorbd_model_path="arm26_with_contact.bioMod", final_time=2, number_shooting_points=20)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    result.graphs()

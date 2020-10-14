import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
from casadi import MX, Function
import pickle
import os
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points

from biorbd_optim import (
    OptimalControlProgram,
    Problem,
    Bounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
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


def states_to_markers(biorbd_model, ocp, states):
    q = states['q']
    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    return markers


if __name__ == "__main__":
    subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '822'

    data_filename = load_data_filename(subject, trial)
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    # --- Load --- #
    # load_name = "Do_822_contact_2_optimal_gravity_N" + str(number_shooting_points)
    # load_ocp_sol_name = load_name + ".bo"
    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
    ocp, sol = OptimalControlProgram.load(load_name + '.bo')

    # --- Get the results --- #
    # states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    # angle = params["gravity_angle"]/np.pi*180
    # print('Number of shooting points: ', number_shooting_points)
    # print('Gravity rotation: ', angle)

    states, controls = Data.get_data(ocp, sol)

    # gravity = biorbd.to_casadi_func("test", ocp.nlp[0]['model'].getGravity)
    # print('Gravity: ', gravity())

    save_variables_name = load_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'states': states, 'controls': controls},#, 'params': params},
                    handle, protocol=3)

    # data_path = '/home/andre/Optimisation/data/' + subject + '/'
    # kalman_path = data_path + 'Q/'
    # q_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'
    # q_ref = loadmat(kalman_path + q_name)['Q2']


    # q = MX.sym("Q", ocp.nlp[0]['model'].nbQ(), 1)
    # qdot = MX.sym("Qdot", ocp.nlp[0]['model'].nbQdot(), 1)
    # qddot = MX.sym("Qddot", ocp.nlp[0]['model'].nbQddot(), 1)
    # tau = MX.sym("Tau", ocp.nlp[0]['model'].nbQddot(), 1)
    # am = biorbd.to_casadi_func("am", ocp.nlp[0]['model'].CalcAngularMomentum, q, qdot, qddot, True)
    # fd = biorbd.to_casadi_func("fd", ocp.nlp[0]['model'].ForwardDynamics, q, qdot, tau)
    #
    # qddot = fd(states['q'], states['q_dot'], controls['tau'])
    # momentum = am(states['q'], states['q_dot'], qddot)

    # from matplotlib import pyplot
    # from matplotlib.lines import Line2D
    #
    # dofs = [range(0, 6), range(6, 9), range(9, 12),
    #         range(12, 14), range(14, 17), range(17, 19), range(19, 21),
    #         range(21, 23), range(23, 26), range(26, 28), range(28, 30),
    #         range(30, 33), range(33, 34), range(34, 36),
    #         range(36, 39), range(39, 40), range(40, 42),
    #         ]
    # dofs_name = ['Pelvis', 'Thorax', 'Head',
    #              'Right shoulder', 'Right arm', 'Right forearm', 'Right hand',
    #              'Left shoulder', 'Left arm', 'Left forearm', 'Left hand',
    #              'Right thigh', 'Right leg', 'Right foot',
    #              'Left thigh', 'Left leg', 'Left foot',
    #              ]
    # # dofs = range(0, 6)
    # for idx_dof, dof in enumerate(dofs):
    #     fig = pyplot.figure()
    #     # pyplot.plot(states_kalman['q'][dof, :].T, color='blue')
    #     pyplot.plot(controls['tau'][dof, :].T, color='red')
    #     # pyplot.plot(states['q'][dof, :].T, color='green')
    #
    #     pyplot.title(dofs_name[idx_dof])
    #     # lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    #     # lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    #     lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    #     pyplot.legend([lm_OE], ['OE'])
    #
    # pyplot.show()

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
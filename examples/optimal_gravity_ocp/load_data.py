import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
from casadi import MX, Function
import pickle
import os

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
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 320
    trial = '821_822_2'

    if subject == 'DoCi':
        if trial == '822':
            c3d_name = 'Do_822_contact_2.c3d'
        elif trial == '44_1':
            c3d_name = 'Do_44_mvtPrep_1.c3d'
        elif trial == '44_2':
            c3d_name = 'Do_44_mvtPrep_2.c3d'
        elif trial == '44_3':
            c3d_name = 'Do_44_mvtPrep_3.c3d'
    elif subject == 'JeCh':
        c3d_name = 'Je_833_1.c3d'
    elif subject == 'BeLa':
        if trial == '44_1':
            c3d_name = 'Ben_44_mvtPrep_1.c3d'
        elif trial == '44_2':
            c3d_name = 'Ben_44_mvtPrep_2.c3d'
        elif trial == '44_3':
            c3d_name = 'Ben_44_mvtPrep_3.c3d'
    elif subject == 'GuSe':
        if trial == '44_2':
            c3d_name = 'Gui_44_mvt_Prep_2.c3d'
        elif trial == '44_3':
            c3d_name = 'Gui_44_mvt_Prep_3.c3d'
        elif trial == '44_4':
            c3d_name = 'Gui_44_mvtPrep_4.c3d'
    elif subject == 'SaMi':
        model_name = 'SaMi.s2mMod'
        if trial == '821_822_2':
            c3d_name = 'Sa_821_822_2.c3d'
        elif trial == '821_822_3':
            c3d_name = 'Sa_821_822_3.c3d'
        elif trial == '821_822_4':
            c3d_name = 'Sa_821_822_4.c3d'
    else:
        raise Exception(subject + ' is not a valid subject')


    # --- Load --- #
    # load_name = "Do_822_contact_2_optimal_gravity_N" + str(number_shooting_points)
    # load_ocp_sol_name = load_name + ".bo"
    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(number_shooting_points) + '_rotated_model' + ".bo"
    ocp, sol = OptimalControlProgram.load(load_name)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    angle = params["gravity_angle"]/np.pi*180
    print('Number of shooting points: ', number_shooting_points)
    print('Gravity rotation: ', angle)

    save_variables_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(number_shooting_points) + '_rotated_model' + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'states': states, 'controls': controls, 'params': params},
                    handle, protocol=3)

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    kalman_path = data_path + 'Q/'
    q_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'
    q_ref = loadmat(kalman_path + q_name)['Q2']


    # q = MX.sym("Q", ocp.nlp[0]['model'].nbQ(), 1)
    # qdot = MX.sym("Qdot", ocp.nlp[0]['model'].nbQdot(), 1)
    # qddot = MX.sym("Qddot", ocp.nlp[0]['model'].nbQddot(), 1)
    # tau = MX.sym("Tau", ocp.nlp[0]['model'].nbQddot(), 1)
    # am = biorbd.to_casadi_func("am", ocp.nlp[0]['model'].CalcAngularMomentum, q, qdot, qddot, True)
    # fd = biorbd.to_casadi_func("fd", ocp.nlp[0]['model'].ForwardDynamics, q, qdot, tau)
    #
    # qddot = fd(states['q'], states['q_dot'], controls['tau'])
    # momentum = am(states['q'], states['q_dot'], qddot)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=number_shooting_points)
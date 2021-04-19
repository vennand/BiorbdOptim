import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function
import pickle
from scipy.io import loadmat
import os
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D

from biorbd_optim import (
    OptimalControlProgram,
    Simulate,
    Problem,
    Bounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
)



def states_to_markers(biorbd_model, states):
    q = states['q']
    n_q = biorbd_model.nbQ()
    n_mark = biorbd_model.nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "markers_func", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    return markers


def states_to_markers_velocity(biorbd_model, states):
    q = states['q']
    qdot = states['q_dot']
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_mark = biorbd_model.nbMarkers()
    n_frames = q.shape[1]

    markers_velocity = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_q = MX.sym("q", n_q, 1)
    symbolic_qdot = MX.sym("qdot", n_qdot, 1)
    # This doesn't work for some mysterious reasons
    # markers_func = Function(
    #     "markers_func", [symbolic_q, symbolic_qdot], [biorbd_model.markersVelocity(symbolic_q, symbolic_qdot)], ["q", "q_dot"], ["markers_velocity"]
    # ).expand()
    for j in range(n_mark):
        markers_func = biorbd.to_casadi_func('markers_func', biorbd_model.markerVelocity, symbolic_q, symbolic_qdot, j)
        for i in range(n_frames):
            markers_velocity[:, j, i] = markers_func(q[:, i], qdot[:, i]).full().squeeze()

    return markers_velocity



if __name__ == "__main__":
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_contact_2'

    data_path = '/home/andre/Optimisation/data/FromGauthier/'
    model_path = data_path + 'Model_2007/'
    c3d_path = data_path + 'c3d/'
    kalman_path = data_path + 'recons/'

    data_filename = 'MultiFP_MOD2007_rightHanded_GenderH_GdFp_'
    E2 = '.E2'
    Q2 = '.Q2'
    data = 'data.mat'
    recons = 'recons.mat'
    model_name = 'Model.s2mMod'
    c3d_name = 'MultiFP.c3d'

    biorbd_model = biorbd.Model(model_path + model_name)
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    frequency = c3d['header']['points']['frame_rate']
    # duration = len(frames) / frequency

    # --- Functions --- #
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)
    am = biorbd.to_casadi_func("am", biorbd_model.CalcAngularMomentum, q, qdot, qddot, True)
    fd = biorbd.to_casadi_func("fd", biorbd_model.ForwardDynamics, q, qdot, tau)
    mcm = biorbd.to_casadi_func("fd", biorbd_model.mass)
    vcm = biorbd.to_casadi_func("fd", biorbd_model.CoMdot, q, qdot)


    # --- Load --- #
    q_kalman = loadmat(kalman_path + data_filename + data)['Q'].squeeze()
    qdot_kalman = loadmat(kalman_path + data_filename + data)['QDot'].squeeze()
    qddot_kalman = loadmat(kalman_path + data_filename + data)['QDDot'].squeeze()

    for q, qdot, qddot in zip(q_kalman, qdot_kalman, qddot_kalman):
        number_shooting_points = q[:, 2:].shape[1]
        duration = number_shooting_points / frequency

        states_kalman = {'q': q[:, 2:], 'q_dot': qdot[:, 2:]}
        controls_kalman = {'tau': id(q[:, 2:], qdot[:, 2:], qddot[:, 2:])}

        markers_kalman = states_to_markers(biorbd_model, states_kalman)
        markers_velocity_EKF_matlab = states_to_markers_velocity(biorbd_model, states_kalman)

        # --- Markers velocity --- #

        model_labels = [label.to_string() for label in biorbd_model.markerNames()]

        norm_markers_velocity_EKF_matlab = np.linalg.norm(markers_velocity_EKF_matlab, axis=0)

        idx_max_all_velocity_OE = np.argmax(norm_markers_velocity_EKF_matlab, axis=0)
        idx_max_velocity_OE = np.unravel_index(norm_markers_velocity_EKF_matlab.argmax(), norm_markers_velocity_EKF_matlab.shape)
        idx_max_middle_velocity_OE = np.unravel_index(
            norm_markers_velocity_EKF_matlab[:, int(number_shooting_points/4):int(number_shooting_points*3/4)].argmax(),
            norm_markers_velocity_EKF_matlab[:, int(number_shooting_points/4):int(number_shooting_points*3/4)].shape)
        idx_min_all_velocity_OE = np.argmin(norm_markers_velocity_EKF_matlab, axis=0)
        idx_min_velocity_OE = np.unravel_index(norm_markers_velocity_EKF_matlab.argmin(), norm_markers_velocity_EKF_matlab.shape)
        idx_min_middle_velocity_OE = np.unravel_index(
            norm_markers_velocity_EKF_matlab[:, int(number_shooting_points/4):int(number_shooting_points*3/4)].argmin(),
            norm_markers_velocity_EKF_matlab[:, int(number_shooting_points/4):int(number_shooting_points*3/4)].shape)

        # --- Plots --- #

        fig_marker_velocity = pyplot.figure(figsize=(20, 10))
        pyplot.plot(norm_markers_velocity_EKF_matlab.T, color='green')

        # dofs = range(0, 36)
        # for dof in dofs:
        #     fig_qdot = pyplot.figure()
        #     pyplot.plot(states_kalman['q_dot'][dof, :].T, color='blue')
        #     pyplot.title(model_labels[dof])


        print('Max velocity for whole movement: ', model_labels[idx_max_velocity_OE[0]], norm_markers_velocity_EKF_matlab[idx_max_velocity_OE])
        print('Max velocity for middle of movement: ', model_labels[idx_max_middle_velocity_OE[0]], norm_markers_velocity_EKF_matlab[:, int(number_shooting_points/4):int(number_shooting_points*3/4)][idx_max_middle_velocity_OE])
        print('Min velocity for whole movement: ', model_labels[idx_min_velocity_OE[0]], norm_markers_velocity_EKF_matlab[idx_min_velocity_OE])
        print('Min velocity for middle of movement: ', model_labels[idx_min_middle_velocity_OE[0]], norm_markers_velocity_EKF_matlab[:, int(number_shooting_points / 4):int(number_shooting_points * 3 / 4)][idx_min_middle_velocity_OE])
        print('Mean max velocity: ', np.mean(norm_markers_velocity_EKF_matlab[idx_max_all_velocity_OE, range(0, idx_max_all_velocity_OE.size)]))
        print('Mean min velocity: ', np.mean(norm_markers_velocity_EKF_matlab[idx_min_all_velocity_OE, range(0, idx_min_all_velocity_OE.size)]))
        print('Mean velocity: ', np.mean(norm_markers_velocity_EKF_matlab))

    pyplot.show()

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
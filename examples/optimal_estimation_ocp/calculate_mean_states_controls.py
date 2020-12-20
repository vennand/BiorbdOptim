import biorbd
import ezc3d
from casadi import MX, Function
import pickle
from scipy.io import loadmat
import numpy as np
import os
import sys
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
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

# essai = {'DoCi': ['822', '44_1', '44_2', '44_3'],
#          'JeCh': ['833_1', '833_2', '833_3', '833_4', '833_5'],
#          'BeLa': ['44_1', '44_2', '44_3'],
#          'GuSe': ['44_2', '44_3', '44_4'],
#          'SaMi': ['821_822_2', '821_822_3',
#                   '821_contact_1', '821_contact_2', '821_contact_3',
#                   '822_contact_1',
#                   '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5']}
essai = {
         'DoCi': ['822']#, '44_1', '44_2', '44_3'],
         # 'JeCh': ['833_1', '833_2', '833_3', '833_4', '833_5'],
         # 'BeLa': ['44_1', '44_3'],
         # 'GuSe': ['44_2', '44_3', '44_4'],
         # 'SaMi': ['821_contact_1', '821_contact_3',
         #          '822_contact_1',
         #          '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5']
}

angular_momentum = np.array(0)
linear_momentum = np.array(0)
for subject, trials in essai.items():
    # print('Subject: ', subject)
    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'
    kalman_path = data_path + 'Q/'
    for trial in trials:
        if (subject == 'BeLa' and trial == '44_2') or \
           (subject == 'GuSe' and trial == '44_2'):
            number_shooting_points = 80
        else:
            number_shooting_points = 100

        data_filename = load_data_filename(subject, trial)
        model_name = data_filename['model']
        c3d_name = data_filename['c3d']
        q_name = data_filename['q']
        qd_name = data_filename['qd']
        qdd_name = data_filename['qdd']
        frames = data_filename['frames']

        biorbd_model = biorbd.Model(model_path + model_name)
        biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
        c3d = ezc3d.c3d(c3d_path + c3d_name)

        frequency = c3d['header']['points']['frame_rate']
        duration = len(frames) / frequency

        # --- Adjust number of shooting points --- #
        adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

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
        load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
        load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
        ocp, sol = OptimalControlProgram.load(load_name + ".bo")

        load_variables_name = load_name + ".pkl"
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        markers_mocap = data['mocap']
        frames = data['frames']
        step_size = data['step_size']

        load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
        optimal_gravity_filename = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".bo"

        q_kalman = loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size]
        qdot_kalman = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
        qddot_kalman = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]

        states_kalman = {'q': q_kalman, 'q_dot': qdot_kalman}
        controls_kalman = {'tau': id(q_kalman, qdot_kalman, qddot_kalman)}

        load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
        with open(load_variables_name, 'rb') as handle:
            kalman_states = pickle.load(handle)
        q_kalman_biorbd = kalman_states['q'][:, ::step_size]
        qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
        qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

        states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
        controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)}

        # --- Get the results --- #
        states, controls = Data.get_data(ocp, sol)
        qddot = fd(states['q'], states['q_dot'], controls['tau'])

        ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
        states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)
        angle = params_optimal_gravity["gravity_angle"].squeeze()
        qddot_optimal_gravity = fd(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], controls_optimal_gravity['tau'])

        rotating_gravity(biorbd_model, angle.squeeze())

        dofs = [range(0, 6), range(6, 9), range(9, 12),
                range(12, 14), range(14, 17), range(17, 19), range(19, 21),
                range(21, 23), range(23, 26), range(26, 28), range(28, 30),
                range(30, 33), range(33, 34), range(34, 36),
                range(36, 39), range(39, 40), range(40, 42),
                ]
        dofs_name = ['Pelvis', 'Thorax', 'Head',
                     'Right shoulder', 'Right arm', 'Right forearm', 'Right hand',
                     'Left shoulder', 'Left arm', 'Left forearm', 'Left hand',
                     'Right thigh', 'Right leg', 'Right foot',
                     'Left thigh', 'Left leg', 'Left foot',
                     ]
        dof_fig_row = [0, 0, 0,
                       2, 2, 2, 2,
                       4, 4, 4, 4,
                       6, 6, 6,
                       8, 8, 8]
        dof_fig_col = [0, 1, 2,
                       0, 1, 2, 3,
                       0, 1, 2, 3,
                       0, 1, 2,
                       0, 1, 2]
        fig, axs = pyplot.subplots(nrows=10, ncols=4, figsize=(11, 9))
        subplots_to_delete = [3, 7, 27, 31, 35, 39]
        # manager = pyplot.get_current_fig_manager()
        # manager.window.showMaximized()
        for subplot in subplots_to_delete:
            fig.delaxes(axs.flatten()[subplot])
        for idx_dof, dof in enumerate(dofs):
            row = dof_fig_row[idx_dof]
            col = dof_fig_col[idx_dof]

            axs[row, col].plot(states_kalman['q'][dof, :].T, color='blue')
            axs[row, col].plot(states_optimal_gravity['q'][dof, :].T, color='red')
            axs[row, col].plot(states['q'][dof, :].T, color='green')

            axs[row+1, col].plot(controls_kalman['tau'][dof, :].T, color='blue')
            axs[row+1, col].plot(controls_optimal_gravity['tau'][dof, :].T, color='red')
            axs[row+1, col].plot(controls['tau'][dof, :].T, color='green')

            # fig = pyplot.figure()
            # pyplot.plot(qdot_ref_matlab[dof, :].T, color='blue')
            # pyplot.plot(qdot_ref_biorbd[dof, :].T, color='red')
            #
            # fig = pyplot.figure()
            # pyplot.plot(qddot_ref_matlab[dof, :].T, color='blue')
            # pyplot.plot(qddot_ref_biorbd[dof, :].T, color='red')

            axs[row, col].set_title(dofs_name[idx_dof], fontsize=10)
            if col == 0:
                axs[row, col].set_ylabel('Joint angles', rotation=0, horizontalalignment='right')
                axs[row+1, col].set_ylabel('Torque', rotation=0, horizontalalignment='right')
            axs[row, col].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[row+1, col].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[row, col].tick_params(axis='y', which='both', labelsize=8)
            axs[row+1, col].tick_params(axis='y', which='both', labelsize=8)

        lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
        lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        fig.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])
        fig.tight_layout()
        fig.subplots_adjust(hspace=1.2, wspace=0.2)

        # fig.set_size_inches((8.5, 11), forward=False)
        save_path = 'Solutions/'
        save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + "_Q_U_N" + str(adjusted_number_shooting_points) + '.png'
        pyplot.savefig(save_name, dpi=500)

    # pyplot.show()

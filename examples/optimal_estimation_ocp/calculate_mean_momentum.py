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
         # 'DoCi': ['822', '44_1', '44_2', '44_3'],
         # 'JeCh': ['833_1', '833_2', '833_3', '833_4', '833_5'],
         # 'BeLa': ['44_1', '44_3'],
         'GuSe': ['44_2', '44_3', '44_4'],
         # 'SaMi': ['821_contact_1', '821_contact_3',
         #          '822_contact_1',
         #          '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5']
}

angular_momentum = np.array(0)
linear_momentum = np.array(0)
for subject, trials in essai.items():
    # print('Subject: ', subject)
    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    c3d_path = data_path + 'Essai/'
    for trial in trials:
        if (subject == 'BeLa' and trial == '44_2') or \
           (subject == 'GuSe' and trial == '44_2'):
            number_shooting_points = 80
        else:
            number_shooting_points = 100

        data_path = '/home/andre/Optimisation/data/' + subject + '/'
        model_path = data_path + 'Model/'
        c3d_path = data_path + 'Essai/'
        kalman_path = data_path + 'Q/'

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

        momentum_OE = am(states['q'], states['q_dot'], qddot).full()[:, 1:]
        momentum_optimal_gravity = am(states_optimal_gravity['q'], states_optimal_gravity['q_dot'], qddot_optimal_gravity).full()[:, 1:]
        momentum_kalman = am(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()[:, 1:]

        total_mass = mcm()['o0'].full()
        linear_momentum_OE = total_mass * vcm(states['q'], states['q_dot']).full()[:, 1:]
        linear_momentum_optimal_gravity = total_mass * vcm(states_optimal_gravity['q'], states_optimal_gravity['q_dot']).full()[:, 1:]
        linear_momentum_kalman = total_mass * vcm(q_kalman_biorbd, qdot_kalman_biorbd).full()[:, 1:]

        slope_lm, _ = np.polyfit(range(linear_momentum_OE.shape[1]), linear_momentum_OE.T, 1) / total_mass / (len(frames) / frequency / adjusted_number_shooting_points)
        slope_lm_optimal_gravity, _ = np.polyfit(range(linear_momentum_optimal_gravity.shape[1]), linear_momentum_optimal_gravity.T, 1) / total_mass / (len(frames) / frequency / adjusted_number_shooting_points)
        slope_lm_kalman, _ = np.polyfit(range(linear_momentum_kalman.shape[1]), linear_momentum_kalman.T, 1) / total_mass / (len(frames) / frequency / adjusted_number_shooting_points)

        fig = pyplot.figure()
        lm_oe = pyplot.plot(momentum_OE.T, color='blue')
        lm_og = pyplot.plot(momentum_optimal_gravity.T, color='orange', linestyle=':')
        lm_kal = pyplot.plot(momentum_kalman.T, color='green')

        lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
        lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        pyplot.legend([lm_oe, lm_og, lm_kal], ['Estimation', 'Optimal gravity', 'Kalman'])
        pyplot.title('Subject: ' + subject + ', Trial: ' + trial + '\nAngular momentum of free fall movement')

        pyplot.annotate('x', (momentum_OE.shape[1] - 1, momentum_OE.T[-1, 0]), textcoords="offset points", xytext=(0, 10), ha='center')
        pyplot.annotate('y', (momentum_OE.shape[1] - 1, momentum_OE.T[-1, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
        pyplot.annotate('z', (momentum_OE.shape[1] - 1, momentum_OE.T[-1, 2]), textcoords="offset points", xytext=(0, 10), ha='center')

        # pyplot.savefig('Angular_momentum_N' + str(number_shooting_points) + '.png')

        fig = pyplot.figure()
        pyplot.plot(linear_momentum_OE.T, color='blue')
        pyplot.plot(linear_momentum_optimal_gravity.T, color='orange', linestyle=':')
        pyplot.plot(linear_momentum_kalman.T, color='green')
        # pyplot.legend(['Estimation', 'Optimal gravity', 'Kalman'])

        lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
        lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
        lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
        pyplot.legend([lm_oe, lm_og, lm_kal], ['OE', 'OGE', 'Kalman'])
        pyplot.title('Subject: ' + subject + ', Trial: ' + trial + '\nLinear momentum of free fall movement')

        pyplot.annotate('x', (linear_momentum_OE.shape[1] - 1, linear_momentum_OE.T[-1, 0]), textcoords="offset points", xytext=(0, 10), ha='center')
        pyplot.annotate('y', (linear_momentum_OE.shape[1] - 1, linear_momentum_OE.T[-1, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
        pyplot.annotate('z', (linear_momentum_OE.shape[1] - 1, linear_momentum_OE.T[-1, 2]), textcoords="offset points", xytext=(0, 10), ha='center')

        box_text = (
                'Kalman gravity norm by linear regression: ' + f"{np.linalg.norm(slope_lm_kalman):.4f}" + '\n'
                'OG gravity norm: ' + f"{np.linalg.norm(slope_lm_optimal_gravity):.4f}" + '\n'
                'OE gravity norm: ' + f"{np.linalg.norm(slope_lm):.4f}"
        )
        text_box = AnchoredText(box_text, frameon=True, loc=3, pad=0.5)
        pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
        pyplot.gca().add_artist(text_box)

        # pyplot.savefig('Linear_momentum_N' + str(adjusted_number_shooting_points) + '.png')

        # if '44' in trial:
        #     times_44.append(duration)
        # if '821' in trial:
        #     times_821.append(duration)
        # if '822' in trial and '821' not in trial:
        #     times_822.append(duration)
        # if '833' in trial:
        #     times_833.append(duration)

# fig = pyplot.figure()
# lm_oe = pyplot.plot(angular_momentum.T, color='blue')
# lm_og = pyplot.plot(momentum_optimal_gravity.T, color='orange', linestyle=':')
# lm_kal = pyplot.plot(momentum_kalman.T, color='green')
#
# lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
# lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
# lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
# pyplot.legend([lm_oe, lm_og, lm_kal], ['Estimation', 'Optimal gravity', 'Kalman'])
# pyplot.title('Subject: ' + subject + ', Trial: ' + trial + '\nAngular momentum of free fall movement')
#
# pyplot.annotate('x', (angular_momentum.shape[1] - 1, angular_momentum.full().T[-1, 0]), textcoords="offset points", xytext=(0, 10), ha='center')
# pyplot.annotate('y', (angular_momentum.shape[1] - 1, angular_momentum.full().T[-1, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
# pyplot.annotate('z', (angular_momentum.shape[1] - 1, angular_momentum.full().T[-1, 2]), textcoords="offset points", xytext=(0, 10), ha='center')
#
# # pyplot.savefig('Angular_momentum_N' + str(number_shooting_points) + '.png')
#
# fig = pyplot.figure()
# pyplot.plot(linear_momentum.T, color='blue')
# pyplot.plot(linear_momentum_optimal_gravity.T, color='orange', linestyle=':')
# pyplot.plot(linear_momentum_kalman.T, color='green')
# # pyplot.legend(['Estimation', 'Optimal gravity', 'Kalman'])
#
# lm_oe = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
# lm_og = Line2D([0, 1], [0, 1], linestyle=':', color='orange')
# lm_kal = Line2D([0, 1], [0, 1], linestyle='-', color='green')
# pyplot.legend([lm_oe, lm_og, lm_kal], ['OE', 'OGE', 'Kalman'])
# pyplot.title('Subject: ' + subject + ', Trial: ' + trial + '\nLinear momentum of free fall movement')
#
# pyplot.annotate('x', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 0]), textcoords="offset points", xytext=(0, 10), ha='center')
# pyplot.annotate('y', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
# pyplot.annotate('z', (linear_momentum.shape[1] - 1, linear_momentum.full().T[-1, 2]), textcoords="offset points", xytext=(0, 10), ha='center')
#
# box_text = (
#         'Kalman gravity norm by linear regression: ' + f"{np.linalg.norm(slope_lm_kalman):.4f}" + '\n'
#         'OG gravity norm: ' + f"{np.linalg.norm(slope_lm_optimal_gravity):.4f}" + '\n'
#         'OE gravity norm: ' + f"{np.linalg.norm(slope_lm):.4f}"
# )
# text_box = AnchoredText(box_text, frameon=True, loc=3, pad=0.5)
# pyplot.setp(text_box.patch, facecolor='white', alpha=0.5)
# pyplot.gca().add_artist(text_box)

pyplot.show()

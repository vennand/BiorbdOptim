import biorbd
import matlab.engine
import numpy as np
from casadi import MX
import ezc3d
import pickle
from scipy.io import loadmat
import os
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from adjust_Kalman import shift_by_2pi


if __name__ == "__main__":
    subjects_trials = [('DoCi', '822', 100), ('DoCi', '44_1', 100), ('DoCi', '44_2', 100), ('DoCi', '44_3', 100),
                     ('BeLa', '44_1', 100), ('BeLa', '44_2', 80), ('BeLa', '44_3', 100),
                     ('GuSe', '44_2', 80), ('GuSe', '44_3', 100), ('GuSe', '44_4', 100),
                     ('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100), ('SaMi', '821_contact_3', 100), ('SaMi', '822_contact_1', 100),
                     ('SaMi', '821_seul_1', 100), ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_3', 100), ('SaMi', '821_seul_4', 100), ('SaMi', '821_seul_5', 100),
                     ('SaMi', '821_822_2', 100), ('SaMi', '821_822_3', 100),
                     ('JeCh', '833_1', 100), ('JeCh', '833_2', 100), ('JeCh', '833_3', 100), ('JeCh', '833_4', 100), ('JeCh', '833_5', 100),
                     ]

    fft_freq = []

    FFT_momentum_OE_sum = []
    FFT_momentum_EKF_biorbd_sum = []
    MNF_momentum_OE_sum = []
    MNF_momentum_EKF_biorbd_sum = []

    for subject_trial in subjects_trials:
        subject = subject_trial[0]
        number_shooting_points = subject_trial[2]
        trial = subject_trial[1]

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

        # --- Adjust number of shooting points --- #
        adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
        print('Subject: ', subject)
        print('Trial: ', trial)
        print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
        print('Node step size: ', step_size)

        frequency = c3d['header']['points']['frame_rate']
        duration = len(frames) / frequency

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

        load_variables_name = load_name + ".pkl"
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        states = data['states']
        controls = data['controls']
        markers_mocap = data['mocap']
        frames = data['frames']
        step_size = data['step_size']

        load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
        optimal_gravity_filename = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'

        optimal_gravity_filename = optimal_gravity_filename + ".pkl"
        with open(optimal_gravity_filename, 'rb') as handle:
            data = pickle.load(handle)

        states_optimal_gravity = data['states']
        controls_optimal_gravity = data['controls']

        q_kalman = shift_by_2pi(biorbd_model, loadmat(kalman_path + q_name)['Q2'][:, frames.start:frames.stop:step_size])
        qdot_kalman = loadmat(kalman_path + qd_name)['V2'][:, frames.start:frames.stop:step_size]
        qddot_kalman = loadmat(kalman_path + qdd_name)['A2'][:, frames.start:frames.stop:step_size]

        states_kalman = {'q': q_kalman, 'q_dot': qdot_kalman}
        controls_kalman = {'tau': id(q_kalman, qdot_kalman, qddot_kalman).full()}

        load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
        with open(load_variables_name, 'rb') as handle:
            kalman_states = pickle.load(handle)
        q_kalman_biorbd = shift_by_2pi(biorbd_model, kalman_states['q'][:, ::step_size])
        qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
        qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

        states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
        controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()}

        # --- Momentum EKF --- #

        momentum_OE = am(states['q'], states['q_dot'], fd(states['q'], states['q_dot'], controls['tau'])).full()
        momentum_EKF = am(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()

        # --- MATLAB --- #
        eng = matlab.engine.start_matlab()

        L = adjusted_number_shooting_points + 1

        signal = matlab.double(np.sum(momentum_OE, axis=0).tolist())
        P2 = np.abs(np.asarray(eng.fft(signal)).squeeze()) / L
        P1 = P2[0:int(L / 2)]
        P1[1:-1] = 2 * P1[1:-1]
        FFT_momentum_OE_sum.append(P1)
        MNF_momentum_OE_sum.append(eng.meanfreq(signal, (frequency / step_size)))

        signal = matlab.double(np.sum(momentum_EKF, axis=0).tolist())
        P2 = np.abs(np.asarray(eng.fft(signal)).squeeze()) / L
        P1 = P2[0:int(L / 2)]
        P1[1:-1] = 2 * P1[1:-1]
        FFT_momentum_EKF_biorbd_sum.append(P1)
        MNF_momentum_EKF_biorbd_sum.append(eng.meanfreq(signal, (frequency / step_size)))

        eng.quit()

        fft_freq.append(frequency / step_size * np.arange(0, int(L/2)) / L)

        print('Angular momentum OE : ', MNF_momentum_OE_sum[-1])
        print('Angular momentum EKF biorbd : ', MNF_momentum_EKF_biorbd_sum[-1])

    # --- Plots --- #

    fig = pyplot.figure()
    pyplot.plot(fft_freq[0], FFT_momentum_EKF_biorbd_sum[0], 's', color='blue', marker='o')
    pyplot.plot(fft_freq[1], FFT_momentum_EKF_biorbd_sum[1], 's', color='blue', marker='X')
    pyplot.axvline(MNF_momentum_EKF_biorbd_sum[0], color='blue', linestyle='-')
    pyplot.axvline(MNF_momentum_EKF_biorbd_sum[1], color='blue', linestyle='--')

    pyplot.plot(fft_freq[0], FFT_momentum_OE_sum[0], 's', color='green', marker='o')
    pyplot.plot(fft_freq[1], FFT_momentum_OE_sum[1], 's', color='green', marker='X')
    pyplot.axvline(MNF_momentum_OE_sum[0], color='green', linestyle='-')
    pyplot.axvline(MNF_momentum_OE_sum[1], color='green', linestyle='--')

    pyplot.xlabel('Hz')
    pyplot.title('Momentum frequency')

    lm_kalman_o = Line2D([0, 1], [0, 1], linestyle='', marker='o', color='blue')
    lm_kalman_X = Line2D([0, 1], [0, 1], linestyle='', marker='X', color='blue')
    lm_OGE_o = Line2D([0, 1], [0, 1], linestyle='', marker='o', color='red')
    lm_OGE_X = Line2D([0, 1], [0, 1], linestyle='', marker='X', color='red')
    lm_OE_o = Line2D([0, 1], [0, 1], linestyle='', marker='o', color='green')
    lm_OE_X = Line2D([0, 1], [0, 1], linestyle='', marker='X', color='green')
    pyplot.legend([lm_kalman_o, lm_kalman_X, lm_OGE_o, lm_OGE_X, lm_OE_o, lm_OE_X], ['EKF', 'EKF', 'OGE', 'OGE torq diff', 'OE', 'OE torq diff'])

    fig = pyplot.figure()
    pyplot.plot(momentum_EKF.T, color='blue')
    pyplot.plot(momentum_OE.T, color='green')

    pyplot.show()
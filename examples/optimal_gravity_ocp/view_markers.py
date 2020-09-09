import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def correct_Kalman(q_ref, qdot_ref, qddot_ref, frequency):
    q_ref[6:, :] = q_ref[6:, :] - ((q_ref[6:, :] / (2*np.pi)).astype(int) * (2*np.pi))
    # qdot_ref[6:, :-1] = (q_ref[6:, 1:] - q_ref[6:, :-1]) * frequency
    # qddot_ref[6:, :-1] = (qdot_ref[6:, 1:] - qdot_ref[6:, :-1]) * frequency

    return q_ref, qdot_ref, qddot_ref

# subject = 'DoCi'
# subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
subject = 'SaMi'
trial = 'bras_volant_2'

data_path = '/home/andre/Optimisation/data/' + subject + '/'
model_path = data_path + 'Model/'
c3d_path = data_path + 'Essai/'
kalman_path = data_path + 'Q/'
if subject == 'DoCi':
    # model_name = 'DoCi.s2mMod'
    model_name = 'DoCi_SystemesDaxesGlobal_surBassin_rotAndre.s2mMod'
    if trial == '822':
        c3d_name = 'Do_822_contact_2.c3d'
        q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
        qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
        qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
        frames = range(3099, 3300)
    elif trial == '44_1':
        c3d_name = 'Do_44_mvtPrep_1.c3d'
        q_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_Q.mat'
        qd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_V.mat'
        qdd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_A.mat'
        frames = range(2450, 2700)
    elif trial == '44_2':
        c3d_name = 'Do_44_mvtPrep_2.c3d'
        q_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_Q.mat'
        qd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_V.mat'
        qdd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_A.mat'
        frames = range(2600, 2850)
    elif trial == '44_3':
        c3d_name = 'Do_44_mvtPrep_3.c3d'
        q_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_Q.mat'
        qd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_V.mat'
        qdd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_A.mat'
        frames = range(4100, 4350)
elif subject == 'JeCh':
    model_name = 'JeCh_201.s2mMod'
    c3d_name = 'Je_833_1.c3d'
    q_name = 'Je_833_1_MOD201.00_GenderM_JeChg_Q.mat'
    qd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_V.mat'
    qdd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_A.mat'
    frames = range(1929, 2200)
elif subject == 'BeLa':
    model_name = 'BeLa_SystemeDaxesGlobal_surBassin.s2mMod'
    if trial == '44_1':
        c3d_name = 'Ben_44_mvtPrep_1.c3d'
        q_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_Q.mat'
        qd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_V.mat'
        qdd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_A.mat'
        frames = range(1799, 2050)
    elif trial == '44_2':
        c3d_name = 'Ben_44_mvtPrep_2.c3d'
        q_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_Q.mat'
        qd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_V.mat'
        qdd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_A.mat'
        frames = range(2149, 2350)
    elif trial == '44_3':
        c3d_name = 'Ben_44_mvtPrep_3.c3d'
        q_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_Q.mat'
        qd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_V.mat'
        qdd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_A.mat'
        frames = range(2450, 2700)
elif subject == 'GuSe':
    model_name = 'GuSe_SystemeDaxesGlobal_surBassin.s2mMod'
    if trial == '44_2':
        c3d_name = 'Gui_44_mvt_Prep_2.c3d'
        q_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_Q.mat'
        qd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_V.mat'
        qdd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_A.mat'
        frames = range(1649, 1850)
    elif trial == '44_3':
        c3d_name = 'Gui_44_mvt_Prep_3.c3d'
        q_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_Q.mat'
        qd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_V.mat'
        qdd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_A.mat'
        frames = range(1699, 1950)
    elif trial == '44_4':
        c3d_name = 'Gui_44_mvtPrep_4.c3d'
        q_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_Q.mat'
        qd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_V.mat'
        qdd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_A.mat'
        frames = range(1599, 1850)
elif subject == 'SaMi':
    model_name = 'SaMi.s2mMod'
    if trial == '821_822_2':
        c3d_name = 'Sa_821_822_2.c3d'
        q_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3280, 3600)
        # frames = range(3660, 3950)
    elif trial == '821_822_3':
        c3d_name = 'Sa_821_822_3.c3d'
        q_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3140, 3450)
    elif trial == '821_822_4':
        c3d_name = 'Sa_821_822_4.c3d'
        q_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3510, 3820)
    elif trial == '821_contact_1':
        c3d_name = 'Sa_821_contact_1.c3d'
        q_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3020, 3330)
    elif trial == '821_contact_2':
        c3d_name = 'Sa_821_contact_2.c3d'
        q_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3570, 3880)
    elif trial == '821_contact_3':
        c3d_name = 'Sa_821_contact_3.c3d'
        q_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3310, 3620)
    elif trial == '822_contact_1':
        c3d_name = 'Sa_822_contact_1.c3d'
        q_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(5010, 5310)
    elif trial == '821_seul_1':
        c3d_name = 'Sa_821_seul_1.c3d'
        q_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3350, 3650)
    elif trial == '821_seul_2':
        c3d_name = 'Sa_821_seul_2.c3d'
        q_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3430, 3740)
    elif trial == '821_seul_3':
        c3d_name = 'Sa_821_seul_3.c3d'
        q_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3210, 3520)
    elif trial == '821_seul_4':
        c3d_name = 'Sa_821_seul_4.c3d'
        q_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(3310, 3620)
    elif trial == '821_seul_5':
        c3d_name = 'Sa_821_seul_5.c3d'
        q_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(2690, 3000)
    elif trial == 'bras_volant_1':
        c3d_name = 'Sa_bras_volant_1.c3d'
        q_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(0, 4657)
    elif trial == 'bras_volant_2':
        c3d_name = 'Sa_bras_volant_2.c3d'
        q_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_Q.mat'
        qd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_V.mat'
        qdd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_A.mat'
        frames = range(0, 3907)
else:
    raise Exception(subject + ' is not a valid subject')

biorbd_model = biorbd.Model(model_path + model_name)
c3d = ezc3d.c3d(c3d_path + c3d_name)
q_ref = loadmat(kalman_path + q_name)['Q2']
qdot_ref = loadmat(kalman_path + qd_name)['V2']
qddot_ref = loadmat(kalman_path + qdd_name)['A2']

frequency = 200
# q_ref, qdot_ref, qddot_ref = correct_Kalman(q_ref, qdot_ref, qddot_ref, frequency)

biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

markers_mocap = c3d['data']['points'][:3, :95, :] / 1000
c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
model_labels = [label.to_string() for label in biorbd_model.markerNames()]
labels_index = [index_c3d for label in model_labels for index_c3d, c3d_label in enumerate(c3d_labels) if label in c3d_label]
markers_reordered = np.zeros((3, len(labels_index), markers_mocap.shape[2]))
for index, label_index in enumerate(labels_index):
    markers_reordered[:, index, :] = markers_mocap[:, label_index, :]


markers = np.ndarray((3, markers_mocap.shape[1], q_ref.shape[1]))
symbolic_states = MX.sym("x", biorbd_model.nbQ(), 1)
markers_func = Function("ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]).expand()
for i in range(markers_mocap.shape[2]):
    markers[:, :, i] = markers_func(q_ref[:, i])

fig = pyplot.figure()
ax = Axes3D(fig)


for frame in frames:#range(markers.shape[2]):
    ax.scatter(markers[0, :, frame], markers[1, :, frame], markers[2, :, frame])
    ax.scatter(markers_reordered[0, :, frame], markers_reordered[1, :, frame], markers_reordered[2, :, frame])
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 2)
    pyplot.pause(0.01)
    pyplot.draw()
    ax.clear()
    print(frame)


import numpy as np
from x_bounds import x_bounds


def check_Kalman(q_ref):
    segments_with_pi_limits = [14, 16, 17, 18, 23, 25, 26, 27, 30, 33, 36, 39]
    hand_segments = [19, 20, 28, 29]
    bool = np.zeros(q_ref.shape)
    bool[4, :] = ((q_ref[4, :] / (np.pi / 2)).astype(int) != 0)
    for (i, j), q in np.ndenumerate(q_ref[6:, :]):
        if i+6 in segments_with_pi_limits:
            bool[i+6, j] = ((q / np.pi).astype(int) != 0)
        elif i+6 in hand_segments:
            bool[i+6, j] = ((q / (3*np.pi/2)).astype(int) != 0)
        else:
            bool[i+6, j] = ((q / (np.pi/2)).astype(int) != 0)
    states_idx_bool = bool.any(axis=1)

    states_idx_range_list = []
    start_index = 0
    broken_dofs = []
    for idx, bool_idx in enumerate(states_idx_bool):
        if bool_idx:
            stop_index = idx
            if idx != start_index:
                states_idx_range_list.append(range(start_index, stop_index))
            start_index = stop_index + 1
            broken_dofs.append(stop_index)
    if bool.shape[0] != start_index:
        states_idx_range_list.append(range(start_index, bool.shape[0]))
    return states_idx_range_list, broken_dofs


def choose_Kalman(q_ref_1, qdot_ref_1, qddot_ref_1, q_ref_2, qdot_ref_2, qddot_ref_2):
    _, broken_dofs_1 = check_Kalman(q_ref_1)
    _, broken_dofs_2 = check_Kalman(q_ref_2)

    q_ref_chosen = q_ref_1
    qdot_ref_chosen = qdot_ref_1
    qddot_ref_chosen = qddot_ref_1

    for dof in broken_dofs_1:
        if dof not in broken_dofs_2:
            q_ref_chosen[dof, :] = q_ref_2[dof, :]
            qdot_ref_chosen[dof, :] = qdot_ref_2[dof, :]
            qddot_ref_chosen[dof, :] = qddot_ref_2[dof, :]

    return q_ref_chosen, qdot_ref_chosen, qddot_ref_chosen


def shift_by_2pi(biorbd_model, q, error_margin=0.35):
    n_q = biorbd_model.nbQ()
    q[4, :] = q[4, :] - ((2 * np.pi) * (np.mean(q[4, :]) / (2 * np.pi)).astype(int))
    for dof in range(6, n_q):
        q[dof, :] = q[dof, :] - ((2 * np.pi) * (np.mean(q[dof, :]) / (2 * np.pi)).astype(int))
        if ((2 * np.pi) * (1 - error_margin)) < np.mean(q[dof, :]) < ((2 * np.pi) * (1 + error_margin)):
            q[dof, :] = q[dof, :] - (2 * np.pi)
        elif ((2 * np.pi) * (1 - error_margin)) < -np.mean(q[dof, :]) < ((2 * np.pi) * (1 + error_margin)):
            q[dof, :] = q[dof, :] + (2 * np.pi)
    return q

def shift_by_pi(q, error_margin):
    if ((np.pi)*(1-error_margin)) < np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q - np.pi
    elif ((np.pi)*(1-error_margin)) < -np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q + np.pi
    return q


def correct_Kalman(biorbd_model, q):
    error_margin = 0.35

    first_dof_segments_with_3DoFs = [6, 9, 14, 23, 30, 36]
    first_dof_segments_with_2DoFs = [12, 17, 19, 21, 26, 28, 34, 40]

    q = shift_by_2pi(biorbd_model, q, error_margin)

    if ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[4, :])) < ((np.pi) * (1 + error_margin)):
        q[3, :] = shift_by_pi(q[3, :], error_margin)
        q[4, :] = -shift_by_pi(q[4, :], error_margin)
        q[5, :] = shift_by_pi(q[5, :], error_margin)

    for dof in first_dof_segments_with_2DoFs:
        if ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof, :])) < ((np.pi) * (1 + error_margin)):
            q[dof, :] = shift_by_pi(q[dof, :], error_margin)
            q[dof+1, :] = -shift_by_pi(q[dof+1, :], error_margin)

    for dof in first_dof_segments_with_3DoFs:
        if (((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof, :])) < ((np.pi) * (1 + error_margin)) or
            ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof+1, :])) < ((np.pi) * (1 + error_margin)) or
            ((np.pi) * (1 - error_margin)) < np.abs(np.mean(q[dof+2, :])) < ((np.pi) * (1 + error_margin))):
            q[dof, :] = shift_by_pi(q[dof, :], error_margin)
            q[dof+1, :] = -shift_by_pi(q[dof+1, :], error_margin)
            q[dof+2, :] = shift_by_pi(q[dof+2, :], error_margin)

    return q

# def shift_by_pi(q):
#     if np.mean(q) > 0:
#         q = q - np.pi
#     else:
#         q = q + np.pi
#     return q
#
# def correct_Kalman(biorbd_model, q):
#     error_margin = 0.35
#     xmin, xmax = x_bounds(biorbd_model)
#
#     q_min = np.amin(q, axis=1)
#     q_max = np.amax(q, axis=1)
#
#     first_dof_segments_with_3DoFs = [6, 9, 14, 23, 30, 36]
#     first_dof_segments_with_2DoFs = [12, 17, 19, 21, 26, 28, 34, 40]
#
#     n_q = biorbd_model.nbQ()
#     q[4, :] = q[4, :] - ((2 * np.pi) * (np.mean(q[4, :]) / (2 * np.pi)).astype(int))
#     for dof in range(6, n_q):
#         q[dof, :] = q[dof, :] - ((2*np.pi) * (np.mean(q[dof, :]) / (2*np.pi)).astype(int))
#         if ((2*np.pi)*(1-error_margin)) < np.mean(q[dof, :]) < ((2*np.pi)*(1+error_margin)):
#             q[dof, :] = q[dof, :] - (2*np.pi)
#         elif ((2*np.pi)*(1-error_margin)) < -np.mean(q[dof, :]) < ((2*np.pi)*(1+error_margin)):
#             q[dof, :] = q[dof, :] + (2*np.pi)
#
#     if q_min[4] < xmin[4] or q_max[4] > xmax[4]:
#         q[3, :] = shift_by_pi(q[3, :])
#         q[4, :] = -shift_by_pi(q[4, :])
#         q[5, :] = shift_by_pi(q[5, :])
#
#     for dof in first_dof_segments_with_2DoFs:
#         if ((q_min[dof] < xmin[dof] or q_max[dof] > xmax[dof]) or
#             (q_min[dof+1] < xmin[dof+1] or q_max[dof+1] > xmax[dof+1])):
#             q[dof, :] = shift_by_pi(q[dof, :])
#             q[dof+1, :] = -shift_by_pi(q[dof+1, :])
#
#     for dof in first_dof_segments_with_3DoFs:
#         if ((q_min[dof] < xmin[dof] or q_max[dof] > xmax[dof]) or
#             (q_min[dof+1] < xmin[dof+1] or q_max[dof+1] > xmax[dof+1]) or
#             (q_min[dof+2] < xmin[dof+2] or q_max[dof+2] > xmax[dof+2])):
#             q[dof, :] = shift_by_pi(q[dof, :])
#             q[dof+1, :] = -shift_by_pi(q[dof+1, :])
#             q[dof+2, :] = shift_by_pi(q[dof+2, :])
#
#     return q


def adjust_Kalman(biorbd_model, subject, trial, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab, q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd):
    q_ref_matlab = correct_Kalman(biorbd_model, q_ref_matlab)
    q_ref_biorbd = correct_Kalman(biorbd_model, q_ref_biorbd)

    if subject == 'DoCi' or subject == 'BeLa' or subject == 'GuSe':
        q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab, q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd)
        q_ref[:6, :] = q_ref_biorbd[:6, :]
        qdot_ref[:6, :] = qdot_ref_biorbd[:6, :]
        qddot_ref[:6, :] = qddot_ref_biorbd[:6, :]
    if subject == 'JeCh':
        q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab)
    if subject == 'SaMi':
        if (trial == '821_seul_2' or trial == '821_seul_3'
                or trial == '821_contact_1' or trial == '821_contact_2' or trial == '821_contact_3'
                or trial == '821_822_4' or trial == '821_822_5'):
            q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd, q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab)
        else:
            q_ref, qdot_ref, qddot_ref = choose_Kalman(q_ref_matlab, qdot_ref_matlab, qddot_ref_matlab, q_ref_biorbd, qdot_ref_biorbd, qddot_ref_biorbd)

    states_idx_range_list, broken_dofs = check_Kalman(q_ref)
    if broken_dofs is not None:
        print('Abnormal Kalman states at DoFs: ', broken_dofs)
        for dof in broken_dofs:
            q_ref[dof, :] = 0
            qdot_ref[dof, :] = 0
            qddot_ref[dof, :] = 0

    return q_ref, qdot_ref, qddot_ref, (states_idx_range_list, broken_dofs)
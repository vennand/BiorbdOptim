import numpy as np
from BiorbdViz import BiorbdViz

subject = 'DoCi'

data_path = '/home/andre/Optimisation/data/' + subject + '/'
model_path = data_path + 'Model/'
model_name = 'DoCi.s2mMod'

b = BiorbdViz(model_path=model_path+model_name)
b.vtk_window.change_background_color((1, 1, 1))
animate_by_hand = -1


if animate_by_hand == 0:
    n_frames = 200
    all_q = np.zeros((b.nQ, n_frames))
    all_q[4, :] = np.linspace(0, np.pi / 2, n_frames)
    b.load_movement(all_q)
    b.exec()
elif animate_by_hand == 1:
    n_frames = 200
    Q = np.zeros((b.nQ, n_frames))
    Q[4, :] = np.linspace(0, np.pi / 2, n_frames)
    i = 0
    while b.vtk_window.is_active:
        b.set_q(Q[:, i])
        i = (i + 1) % n_frames
else:
    b.exec()
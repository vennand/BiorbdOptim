import numpy as np
import os
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText


if __name__ == "__main__":
    trials = [('GuSe', '44_2'), ('SaMi', '821_seul_5'), ('DoCi', '822_contact'), ('JeCh', '833_1')]

    induced_gravity = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25])

    marker_error = np.array([[23.43, 23.51, 24.10, 25.17, 26.60, 28.41, 40.88, 56.24, 73.51, 91.73],
                             [33.83, 34.16, 36.95, 41.59, 47.30, np.nan, np.nan, np.nan, np.nan, np.nan],
                             [20.05, 21.07, 23.75, 27.47, 31.78, 36.49, 63.67, 94.66, np.nan, np.nan],
                             [33.26, 33.95, 36.41, 40.34, 45.33, 51.02, 84.85, np.nan, np.nan, np.nan]])

    # marker_SD = np.array([[13.11, 13.37, 14.35, 15.93, 17.96, 20.30, 34.00, 49.21, 64.59, 80.04],
    #                       [26.36, 27.57, 30.20, 34.12, 39.07, np.nan, np.nan, np.nan, np.nan, np.nan],
    #                       [12.89, 14.20, 17.34, 21.69, 26.80, 32.35, 62.00, 92.69, np.nan, np.nan],
    #                       [19.10, 20.18, 22.72, 26.34, 30.80, 35.90, 66.32, np.nan, np.nan, np.nan]])


    # --- Plots --- #

    fig = pyplot.figure()

    pyplot.plot(induced_gravity, marker_error.T, 's')
    pyplot.xlabel('Induced gravity deviation (°)')
    pyplot.ylabel('Marker error (mm)')
    fig.legend(['One somersault, two twist', 'Two somersaults, one and a half twists', 'Two somersaults, two twists', 'Two somersaults, three twists'])

    pyplot.show()

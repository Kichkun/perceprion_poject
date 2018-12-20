import numpy as np
from matplotlib import pyplot as plt

from sam.tools.data import load_odometry_and_observations
from sam.tools.plot import *
from sam.SAM import *


def main():
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    take_each = 10

    alphas = np.array([0.06, 0.0015, 0.06, 0.02]) ** 2
    Q = np.diag([10., np.deg2rad(10.), 0]) ** 2

    num_steps, sam_input, sam_debug = load_odometry_and_observations('odometry.npz', 'means_corresp_new.npz', alphas, take_each)

    intial_state = sam_debug.noise_free_robot_path[0]
    initial_sigma = 1e-12 * np.eye(3, 3)

    filter = SAM(intial_state, initial_sigma, alphas, Q)

    for t in range(1, num_steps):
        # Draw field, landmarks and paths
        plt.cla()
        plot_field(sam_debug.real_landmarks_positions, t, sam_debug.visibility_matrix, take_each)
        plot_robot(sam_debug.real_robot_path[t])

        plt.plot(sam_debug.real_robot_path[:t, 0], sam_debug.real_robot_path[0:t, 1], 'r', linewidth=0.5)
        plt.plot([sam_debug.real_robot_path[t, 0]], [sam_debug.real_robot_path[t, 1]], '*r')

        plt.plot(sam_debug.noise_free_robot_path[:t, 0], sam_debug.noise_free_robot_path[:t, 1], 'g', linewidth=0.5)
        plt.plot([sam_debug.noise_free_robot_path[t, 0]], [sam_debug.noise_free_robot_path[t, 1]], '*g')

        # Filter trajectory
        filter.predict(sam_input.motion_commands[t - 1])
        filter.update(sam_input.observations[t])

        # Draw filtered trajectory
        plt.plot(filter.get_robot_position()[:t, 0], filter.get_robot_position()[0:t, 1], 'b', linewidth=0.5)
        plt.plot([filter.get_robot_position()[t, 0]], [filter.get_robot_position()[t, 1]], '*b')

        # Draw landmarks
        plt.scatter(filter.get_landmarks()[:,0], filter.get_landmarks()[:, 1], s=10)

        plt.draw()
        plt.pause(0.01)

    plt.show(block=True)


if __name__ == '__main__':
    main()

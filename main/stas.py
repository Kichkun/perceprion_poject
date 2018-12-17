import numpy as np


def wrap_angle(alpha):
    while alpha >= np.pi:
        alpha -= 2 * np.pi
    while alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def trajectory_to_odometry(trajectory):
    """
    :param trajectory: xs and ys
    :return:
    xs:      xs, ys, bearing;
    motions: rot1, trans, rot2
    """
    xs = np.zeros((trajectory.shape[0], 3), dtype='float')
    motions = np.zeros((trajectory.shape[0] - 1, 3), dtype='float')
    xs[:, :2] = trajectory
    dx0 = trajectory[1, 0] - trajectory[0, 0]
    dy0 = trajectory[1, 1] - trajectory[0, 1]
    xs[0, 2] = np.arctan2(dy0, dx0)
    for i in range(1, trajectory.shape[0]):
        dx = trajectory[i, 0] - trajectory[i - 1, 0]
        dy = trajectory[i, 1] - trajectory[i - 1, 1]
        new_theta = np.arctan2(dy, dx)
        xs[i, 2] = new_theta
        theta = xs[i - 1, 2]
        dtheta = wrap_angle(new_theta - theta)
        motions[i - 1, :] = [dtheta, dx ** 2 + dy ** 2, 0]

    return xs, motions

import numpy as np


def wrap_angle(alpha):
    while alpha >= np.pi:
        alpha -= 2 * np.pi
    while alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def project_traj(t):
    return t[:, :2]


def trajectory_to_odometry(trajectory, take_each):
    """
    :param trajectory: xs and ys
    :return:
    xs:      xs, ys, bearing;
    motions: rot1, trans, rot2
    """
    xs = np.zeros((0, 3), dtype='float')
    motions = np.zeros((0, 3), dtype='float')

    x = np.zeros((1, 3))
    x[0, :2] = trajectory[0]
    x[0, 2] = np.arctan2(trajectory[take_each, 1] - trajectory[0, 1],
                         trajectory[take_each, 0] - trajectory[0, 0])
    xs = np.vstack((xs, x))

    for i in range(take_each, trajectory.shape[0], take_each):
        dx = trajectory[i, 0] - trajectory[i - take_each, 0]
        dy = trajectory[i, 1] - trajectory[i - take_each, 1]

        new_theta = np.arctan2(dy, dx)
        theta = xs[-1, 2]

        x = np.zeros((1, 3))
        x[0, :2] = trajectory[i]
        x[0, 2] = new_theta
        xs = np.vstack((xs, x))

        u = np.array([wrap_angle(new_theta - theta), np.sqrt(dx ** 2 + dy ** 2), 0])
        motions = np.vstack((motions, u))

    print(xs.shape)
    print(motions.shape)

    print(xs[:10])
    print(motions[:10])

    return xs, motions


def main():
    trajectory = project_traj(np.load('trajectory.npy'))
    take_each = 10
    path, motions = trajectory_to_odometry(trajectory, take_each)
    np.savez('odometry', num_steps=500 // take_each, noise_free_motion=motions, noise_free_robot_path=path)


if __name__ == '__main__':
    main()

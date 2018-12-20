import numpy as np
from matplotlib import pyplot as plt


def plot_robot(state, radius=5.):
    """
    Plots a circle at the center of the robot and a line to depict the yaw.

    :param state: numpy.ndarray([x, y, theta]).
    :param radius: The radius of the circle representing the robot.
    """

    assert isinstance(state, np.ndarray)
    assert state.shape == (3,)

    robot = plt.Circle(state[:-1], radius, edgecolor='black', facecolor='cyan', alpha=0.25)
    orientation_line = np.array([[state[0], state[0] + (np.cos(state[2]) * (radius * 1.5))],
                                 [state[1], state[1] + (np.sin(state[2]) * (radius * 1.5))]])

    plt.gcf().gca().add_artist(robot)
    plt.plot(orientation_line[0], orientation_line[1], 'black')


def plot_field(real_landmarks_positions, t, visibility_matrix, coeff):
    """
    Plots the field and highlights the currently detected marker.
    """

    plt.axis((-163, 163, -276, 40))
    plt.xlabel('X')
    plt.ylabel('Y')

    for index, i in enumerate(real_landmarks_positions):
        if visibility_matrix[index, t * coeff] == 1:
            landmark = plt.Circle(i, 1, edgecolor='red', facecolor='red', linewidth=0.5)
        else:
            landmark = plt.Circle(i, 1, edgecolor='black', facecolor='black', linewidth=0.5)

        plt.gcf().gca().add_artist(landmark)
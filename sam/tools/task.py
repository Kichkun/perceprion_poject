import numpy as np
from numpy.random import normal as sample1d


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle


def get_prediction(state, motion):
    """
    Predicts the next state given state and the motion command.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param motion: The motion command to execute (format: [drot1, dtran, drot2]).
    :return: The next state of the robot after executing the motion command
             (format: np.array([x, y, theta])). The angle will be in range
             [-pi, +pi].
    """

    assert state.shape == (3,)
    assert motion.shape == (3,)

    x, y, theta = state
    drot1, dtran, drot2 = motion

    theta += drot1
    x += dtran * np.cos(theta)
    y += dtran * np.sin(theta)
    theta += drot2

    # Wrap the angle between [-pi, +pi].
    theta = wrap_angle(theta)

    return np.array([x, y, theta])


def get_observation(state, landmark, index):
    """
    Generates a sample observation given the current state of the robot and the marker id of which to observe.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param landmark: Observed landmark
    :param index: The landmark id indexing into the landmarks list in the field map.
    :return: The observation to the landmark (format: np.array([range, bearing, landmark_id])).
             The bearing (in rad) will be in [-pi, +pi].
    """

    assert state.shape == (3,)

    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, wrap_angle(bearing), index])


def get_expected_observation(state, m_j):
    dx = m_j[0] - state[0]
    dy = m_j[1] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, bearing])


def h_inv(state, z):
    angle = z[1] + state[2]
    return [state[0] + z[0] * np.cos(angle), state[1] + z[0] * np.sin(angle)]


def apply_noise_to_motion(motion, alphas):
    a1, a2, a3, a4 = alphas
    drot1, dtran, drot2 = motion

    noisy_motion = np.zeros(motion.size)

    noisy_motion[0] = sample1d(drot1, np.sqrt(a1 * (drot1 ** 2) + a2 * (dtran ** 2)))
    noisy_motion[1] = sample1d(dtran, np.sqrt(a3 * (dtran ** 2) + a4 * ((drot1 ** 2) + (drot2 ** 2))))
    noisy_motion[2] = sample1d(drot2, np.sqrt(a1 * (drot2 ** 2) + a2 * (dtran ** 2)))

    return noisy_motion


def sample_from_odometry(state, motion, alphas):
    noised_motion = apply_noise_to_motion(motion, alphas)
    return get_prediction(state, noised_motion)


def Gt(state, u):
    return np.array([[1, 0, -1 * u[1] * np.sin(state[2])],
                     [0, 1, u[1] * np.cos(state[2])],
                     [0, 0, 1]])


def Ht(state, m_j):
    dx = m_j[0] - state[0]
    dy = m_j[1] - state[1]
    q = dx ** 2 + dy ** 2
    sqrt_q = np.sqrt(q)

    return np.array([[-dx / sqrt_q, -dy / sqrt_q, 0],
                     [dy / q, -dx / q, -1]])


def Jt(state, m_j):
    dx = m_j[0] - state[0]
    dy = m_j[1] - state[1]
    q = dx ** 2 + dy ** 2
    sqrt_q = np.sqrt(q)

    return np.array([[dx / sqrt_q, dy / sqrt_q],
                     [-dy / q, dx / q]])


def get_motion_noise_covariance(motion, alphas):
    """
    :param motion: The action command at the current time step (format: [drot1, dtran, drot2]).
    :param alphas: The action noise parameters (format [a1, a2, a3, a4]).
    :return: The covariance of the transition function noise (in action space).
    """

    assert isinstance(motion, np.ndarray)
    assert isinstance(alphas, np.ndarray)

    assert motion.shape == (3,)
    assert alphas.shape == (4,)

    drot1, dtran, drot2 = motion
    a1, a2, a3, a4 = alphas

    return np.diag([a1 * drot1 ** 2 + a2 * dtran ** 2,
                    a3 * dtran ** 2 + a4 * (drot1 ** 2 + drot2 ** 2),
                    a1 * drot2 ** 2 + a2 * dtran ** 2])

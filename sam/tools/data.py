import numpy as np
from numpy.random import multivariate_normal as sample2d

from sam.tools.objects import *
from sam.tools.task import *


def sense_landmarks(real_robot_path, real_landmarks_positions, visibility_matrix, take_each):
    noised_observations = []

    for pos_index, i in enumerate(real_robot_path):
        ith_obs = []
        for obs_index, j in enumerate(visibility_matrix[:, pos_index * take_each]):
            if j == 1:
                ith_obs.append(get_observation(i, real_landmarks_positions[obs_index], obs_index))

        noised_observations.append(np.array(ith_obs))

    return noised_observations


def load_odometry_and_observations(odometry_filename, observation_filename, alphas, take_each):
    """
    :param odometry_filename:
    :param observation_filename:
    :param alphas: motion covariance coefficients
    :param Q: observation covariance
    """

    odometry_data = np.load(odometry_filename)
    observation_data = np.load(observation_filename)

    num_steps = odometry_data['num_steps']

    # Our real path is the ground truth
    real_robot_path = odometry_data['noise_free_robot_path']
    # And motions that lead to this path
    real_motion = odometry_data['noise_free_motion'][:num_steps - 1, :]
    # Real positions of landmarks
    real_landmarks_positions = observation_data['mean'][:, :2]
    # Landmarks observed at each time step
    visibility_matrix = observation_data['corresp']

    robot_state_dim = 3
    motion_dim = 3
    landmark_state_dim = 2

    # We make assumption that our real and noiseless data differs on N(0,R)
    noise_free_robot_path = np.zeros((num_steps, robot_state_dim))
    noise_free_motion_commands = np.zeros((num_steps - 1, motion_dim))

    # Create observations
    noised_observations = sense_landmarks(real_robot_path, real_landmarks_positions, visibility_matrix, take_each)

    noise_free_robot_path[0] = real_robot_path[0]

    for i in range(1, num_steps):
        noise_free_motion_commands[i - 1] = apply_noise_to_motion(real_motion[i - 1], alphas)
        noise_free_robot_path[i] = get_prediction(noise_free_robot_path[i - 1], noise_free_motion_commands[i - 1])

    sam_input = SAMInputData(noise_free_motion_commands, noised_observations)
    sam_debug = SAMDebugData(real_robot_path, noise_free_robot_path, real_landmarks_positions, visibility_matrix)

    return num_steps, sam_input, sam_debug

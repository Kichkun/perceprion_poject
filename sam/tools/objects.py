import numpy as np


class SAMInputData(object):
    """
    Represents the data that is available to the SAM algorithm while estimating the robot state.
    """

    def __init__(self, motion_commands, observations):
        """
        Sets the internal data available to SLAM algorithm while estimating the world state.

        Let M be the number of observations sensed per time step in the simulation.
        Let N be the number of steps in the robot state estimation simulation.

        :param motion_commands: A 2-D numpy ndarray of size Nx3 where each row is [drot1, dtran, drot2].
        :param observations:
        """

        if not isinstance(motion_commands, np.ndarray):
            raise TypeError('motion_commands should be of type np.ndarray.')

        if motion_commands.ndim != 2 or motion_commands.shape[1] != 3:
            raise ValueError('motion_commands should be of size Nx3 where N is the number of time steps.')

        self.motion_commands = motion_commands
        self.observations = observations


class SAMDebugData(object):
    """
    Contains data only available for debugging/displaying purposes during robot state estimation.
    """

    def __init__(self, real_robot_path, noise_free_robot_path, real_landmarks_positions, visibility_matrix):
        """
        Sets the internal data only available for debugging purposes to the state estimation filter.

        Let M be the number of observations sensed per time step in the simulation.
        Let N be the number of steps in the robot state estimation simulation.

        :param real_robot_path: A 2-D numpy ndarray of size Nx3 where each row is [x, y, theta].
        :param noise_free_robot_path: A 2-D numpy ndarray of size Nx3 where each row is [x, y, theta].
        """

        if real_robot_path.ndim != 2 or real_robot_path.shape[1] != 3:
            raise ValueError('real_robot_path should be of size Nx3 where N is the number of time steps.')

        if noise_free_robot_path.ndim != 2 or noise_free_robot_path.shape[1] != 3:
            raise ValueError('noise_free_robot_path should be of size Nx3 where N is the number of time steps.')

        self.real_robot_path = real_robot_path
        self.noise_free_robot_path = noise_free_robot_path
        self.real_landmarks_positions = real_landmarks_positions
        self.visibility_matrix = visibility_matrix

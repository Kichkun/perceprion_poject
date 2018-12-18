import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numpy import dot
from numpy.linalg import inv, norm, svd, solve
from mpl_toolkits.mplot3d import Axes3D

import pykitti
import open3d


def get_trajectory(dataset, max_range):
    trajectory = np.zeros((0, 3))

    # Collect odometry
    for i in np.arange(0, max_range):
        curr = dataset.poses[i][:, 3]
        trajectory = np.vstack((trajectory, curr[:3]))

    return trajectory


def get_point_cloud(dataset, pcs_step, take_each_n_point, max_range):
    pc_s = np.zeros((0, 3))
    # Collect point clouds
    for i in np.arange(0, max_range, pcs_step):
        curr = dataset.poses[i][:, 3]

        # Select n points from i-th point cloud
        pc = dataset.get_velo(i)
        pc = pc[range(0, pc.shape[0], take_each_n_point)]

        # Transform from velodyne (X) to left camera coordinates (x): x = Tr * X
        Tr = dataset.calib.T_cam0_velo[:3, :]
        pc = np.array([dot(Tr, i) for i in pc])
        pc += curr[:3]

        pc_s = np.vstack((pc_s, pc))

    return pc_s


def get_landmarks_mean(landmarks_data):
    return landmarks_data['mean']


def calculate_rotation_matrix(a, b):
    v = np.cross(a, b)
    c = np.dot(a, b)
    I = np.identity(3)

    vx = np.matrix([0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0]).reshape((3, 3))

    return I + vx + np.matmul(vx, vx) / (1 + c)


def find_best_fitting_plane_and_rotate_trajectory(trajectory):
    # First use SVD to find normal of plane and calculate transformation to z-oriented view
    # Center all points
    centroid = [trajectory[:, 0].sum() / trajectory.shape[0],
                trajectory[:, 1].sum() / trajectory.shape[0],
                trajectory[:, 2].sum() / trajectory.shape[0]]

    centered_trajectory = trajectory - centroid

    # Apply SVD
    u, s, vh = svd(centered_trajectory.T, full_matrices=True)

    # Normal
    normal = u[:, 2]

    # Applied in a way: dot(point, M)
    M = calculate_rotation_matrix(np.array([0, 0, 1]), normal)

    normal = np.array(*dot(normal, M).tolist())

    rotated_centroid = np.array(*dot(centroid, M).tolist())
    rotated_trajectory = centered_trajectory[:, :] * M + rotated_centroid

    base = np.array([0.0, 0.0, 0.0]) + rotated_centroid

    normal += rotated_centroid

    # Use least squares to find best fitting plane along z coordinate

    A = np.zeros((rotated_trajectory.shape[0], 3))
    A[0:, 0] = rotated_trajectory[0:, 0].reshape((rotated_trajectory.shape[0],))
    A[0:, 1] = rotated_trajectory[0:, 1].reshape((rotated_trajectory.shape[0],))
    A[0:, 2] = 1

    b = np.zeros((rotated_trajectory.shape[0], 1))
    b[0:, 0] = rotated_trajectory[0:, 2].reshape((rotated_trajectory.shape[0],))

    # ax + by + c = z
    # Here: a, b and c
    plane_coeffs = solve(dot(A.T, A), dot(A.T, b))

    return rotated_trajectory, centroid, rotated_centroid, base, normal, M, plane_coeffs


def rotate_point_clouds(pc_s, centroid, M, rotated_centroid):
    pc_s -= centroid

    pc_s = pc_s[:, :] * M

    pc_s += rotated_centroid

    return pc_s


def compute_z(X, Y, plane_coeffs):
    return plane_coeffs[0] * X + plane_coeffs[1] * Y + plane_coeffs[2]


def project_landmarks_mean(landmarks_mean, plane_coeffs):
    proj_landmarks_mean = np.zeros(landmarks_mean.shape)

    for index, i in enumerate(landmarks_mean):
        proj_landmarks_mean[index] = np.array([*i[:2], compute_z(i[0], i[1], plane_coeffs)])

    return proj_landmarks_mean


def project_noise_free_robot_path(noise_free_robot_path, plane_coeffs):
    proj_noise_free_robot_path = np.zeros(noise_free_robot_path.shape)

    for index, i in enumerate(noise_free_robot_path):
        proj_noise_free_robot_path[index] = np.array([*i[:2], compute_z(i[0], i[1], plane_coeffs)])

    return proj_noise_free_robot_path


def plot_trajectory(rotated_trajectory, ax):
    ax.scatter(rotated_trajectory[:, 0],
               rotated_trajectory[:, 1],
               rotated_trajectory[:, 2],
               c='red')


def plot_plane(base, normal, rotated_centroid, plane_coeffs, ax):
    x_shift = rotated_centroid[0] * 1.4
    y_shift = rotated_centroid[1] * 1.4

    lx = rotated_centroid[0] - x_shift
    ly = rotated_centroid[1] - y_shift
    ux = rotated_centroid[0] + x_shift
    uy = rotated_centroid[1] + y_shift
    num = 10
    ax.plot([base[0], normal[0]], [base[1], normal[1]], [base[2], normal[2]])

    X, Y = np.meshgrid(np.arange(lx, ux, (ux - lx) / num), np.arange(ly, uy, (uy - ly) / num))
    Z = compute_z(X, Y, plane_coeffs)
    ax.plot_wireframe(X, Y, Z, color='dimgray')


def plot_point_clouds(pc_s, ax2):
    ax2.scatter(pc_s[:, 0],
                pc_s[:, 1],
                pc_s[:, 2])


def plot_landmarks_mean(lm, ax2):
    ax2.scatter(lm[:, 0],
                lm[:, 1],
                lm[:, 2])


def plot_proj_landmarks_mean(p_lm, ax2):
    ax2.scatter(p_lm[:, 0],
                p_lm[:, 1],
                p_lm[:, 2])


def plot_proj_noise_free_robot_path(proj_noise_free_robot_path, ax2):
    ax2.scatter(proj_noise_free_robot_path[:, 0],
                proj_noise_free_robot_path[:, 1],
                proj_noise_free_robot_path[:, 2])


def main():
    np.set_printoptions(precision=4, suppress=True)

    # Folder with data set
    basedir = '/Users/apple/Downloads/perception_project-master/dataset'
    # Sequence to use
    sequence = '00'
    # Amount of frames to download.
    max_range = 500
    # How frequently should we select point clouds
    pcs_step = 40
    # Get n points from each of point clouds
    take_each_n_point = 700

    # Load odometry and point clouds
    dataset = pykitti.odometry(basedir, sequence, frames=range(0, max_range, 1))
    landmarks_data = np.load("means_r_3.npz")
    odometry_data = np.load("odometry.npz")

    # Extract needed data
    trajectory = get_trajectory(dataset, max_range)
    # pc_s = get_point_cloud(dataset, pcs_step, take_each_n_point, max_range)
    landmarks_mean = get_landmarks_mean(landmarks_data)
    noise_free_robot_path = odometry_data['noise_free_robot_path']

    rotated_trajectory, centroid, rotated_centroid, base, normal, M, plane_coeffs = find_best_fitting_plane_and_rotate_trajectory(
        trajectory)

    # rotated_pc_s = rotate_point_clouds(pc_s, centroid, M, rotated_centroid)

    proj_landmarks_mean = project_landmarks_mean(landmarks_mean, plane_coeffs)
    proj_noise_free_robot_path = project_noise_free_robot_path(noise_free_robot_path, plane_coeffs)

    # Plot this
    f2 = plt.figure()
    ax2 = f2.add_subplot(111, projection='3d')

    plot_trajectory(rotated_trajectory, ax2)
    # plot_point_clouds(rotated_pc_s, ax2)
    plot_plane(base, normal, rotated_centroid, plane_coeffs, ax2)
    plot_landmarks_mean(landmarks_mean, ax2)

    plot_proj_landmarks_mean(proj_landmarks_mean, ax2)
    plot_proj_noise_free_robot_path(proj_noise_free_robot_path, ax2)

    plt.show()


if __name__ == "__main__":
    main()

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numpy import dot
from numpy.linalg import inv, norm, svd, solve
from mpl_toolkits.mplot3d import Axes3D

import pykitti
import open3d


def draw_point_cloud(dataset, pcs_step, take_each_n_point, ax):
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

    # ax.scatter(pc_s[:, 0],
    #            pc_s[:, 2],
    #            pc_s[:, 1])


def get_trajectory(dataset):
    trajectory = np.zeros((0, 3))

    # Collect odometry
    for i in np.arange(0, max_range):
        curr = dataset.poses[i][:, 3]
        trajectory = np.vstack((trajectory, curr[:3]))

    return trajectory


def compute_z(X, Y, plane_coeffs):
    return plane_coeffs[0] * X + plane_coeffs[1] * Y + plane_coeffs[2]


def calculate_rotation_matrix(a, b):
    v = np.cross(a, b)
    c = np.dot(a, b)
    I = np.identity(3)

    vx = np.matrix([0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0]).reshape((3, 3))

    return I + vx + np.matmul(vx, vx) / (1 + c)


def find_best_fitting_plane(trajectory, ax):
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

    rotated_trajectory = centered_trajectory[:, :] * M

    trajectory = rotated_trajectory + centroid

    # Use least squares to find best fitting plane along z coordinate

    A = np.zeros((trajectory.shape[0], 3))
    A[0:, 0] = trajectory[0:, 0].reshape((trajectory.shape[0],))
    A[0:, 1] = trajectory[0:, 1].reshape((trajectory.shape[0],))
    A[0:, 2] = 1

    b = np.zeros((trajectory.shape[0], 1))
    b[0:, 0] = trajectory[0:, 2].reshape((trajectory.shape[0],))

    # ax + by + c = z
    # Here: a, b and c
    plane_coeffs = solve(dot(A.T, A), dot(A.T, b))

    lx = -10
    ly = -130
    ux = 90
    uy = 200
    num = 10

    base = np.array([0, 0, 0]) + centroid
    normal += centroid

    ax.plot([base[0], normal[0]], [base[1], normal[1]], [base[2], normal[2]])

    X, Y = np.meshgrid(np.arange(lx, ux, (ux - lx) / num), np.arange(ly, uy, (uy - ly) / num))
    Z = compute_z(X, Y, plane_coeffs)

    ax.plot_wireframe(X, Y, Z, color='dimgray')

    return trajectory


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
take_each_n_point = 200

# Load odometry and point clouds
dataset = pykitti.odometry(basedir, sequence, frames=range(0, max_range, 1))

# Plot this
f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')

# draw_point_cloud(dataset, pcs_step, take_each_n_point, ax2)
trajectory = get_trajectory(dataset)

trajectory = find_best_fitting_plane(trajectory, ax2)

ax2.scatter(trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            c='red')

plt.show()

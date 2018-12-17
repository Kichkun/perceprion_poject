import copy

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numpy import dot
from numpy.linalg import inv, norm, svd, solve
from mpl_toolkits.mplot3d import Axes3D

import pykitti

from open3d.open3d import (
    estimate_normals, KDTreeSearchParamHybrid,
    compute_fpfh_feature, read_point_cloud,
    registration_ransac_based_on_feature_matching,
    TransformationEstimationPointToPoint,
    CorrespondenceCheckerBasedOnEdgeLength,
    CorrespondenceCheckerBasedOnDistance, RANSACConvergenceCriteria,
    registration_icp, TransformationEstimationPointToPlane,
    draw_geometries, voxel_down_sample
)


def calculate_global_pcd(dataset, numb, take_each_n_point, ax, centroids):
    pc_s = np.zeros((0, 3))
    # Collect point clouds
    for i in (np.arange(0, numb, pcs_step)):
        curr = dataset.poses[i][:, 3]

        # Select n points from i-th point cloud
        pc = dataset.get_velo(i)
        pc = pc[range(0, pc.shape[0], take_each_n_point)]

        # Transform from velodyne (X) to left camera coordinates (x): x = Tr * X
        Tr = dataset.calib.T_cam0_velo[:3, :]
        pc = np.array([dot(Tr, i) for i in pc])
        pc += curr[:3]

        pc_s = np.vstack((pc_s, pc))
        ax.scatter(pc[:, 0],
                   pc[:, 2],
                   pc[:, 1])  # , '.g')'''
        print(pc * centroids)


def to_normal_axes(points):
    """
    in kitti z axis mean forward and y - up, this function swaps them
    :param points: Nx3
    :return: Nx3
    """
    res = np.empty_like(points)
    res[:, 0] = points[:, 0]
    res[:, 1] = points[:, 2]
    res[:, 2] = points[:, 1]
    return res


def draw_point_cloud(dataset, pcs_step, take_each_n_point, ax, max_range):
    pc_s = np.zeros((0, 3))
    # Collect point clouds
    for i in np.arange(0, max_range, pcs_step):
        curr = dataset.poses[i][:3, 3]

        # Select n points from i-th point cloud
        pc = dataset.get_velo(i)
        pc = pc[::take_each_n_point]

        # Transform from velodyne (X) to left camera coordinates (x): x = Tr * X
        pc = (dataset.calib.T_cam0_velo @ pc.T).T
        pc = pc[:, :3]
        pc += curr
        pc = to_normal_axes(pc)
        pc_s = np.vstack((pc_s, pc))

    ax.scatter(
        pc_s[:, 0],
        pc_s[:, 1],
        pc_s[:, 2],
        s=2
    )


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
    print(centroid)
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

    return trajectory, centroid


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])


def draw_registration_result_part(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    estimate_normals(pcd_down, KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = compute_fpfh_feature(pcd_down,
                                    KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = read_point_cloud("000000.pcd")
    target = read_point_cloud("000099.pcd")
    '''trans_init = np.asarray(    [[0.862, 0.011, -0.507, 0.5],
     [-0.139, 0.967, -0.215, 0.7],
     [0.487, 0.255, 0.835, -1.4],
     [0.0, 0.0, 0.0, 1.0]])'''
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        distance_threshold,
        TransformationEstimationPointToPoint(False), 4,
        [CorrespondenceCheckerBasedOnEdgeLength(0.9),
         CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = registration_icp(source, target, distance_threshold,
                              result_ransac.transformation,
                              TransformationEstimationPointToPlane())
    return result


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Folder with data set
    basedir = '/Users/s.tsepa/Workspace/prob-rob/proj/perception_project/dataset'
    # Sequence to use
    sequence = '00'
    # Amount of frames to download.
    max_range = 200
    # How frequently should we select point clouds
    pcs_step = 25
    # Get n-th points from each of point clouds
    take_each_n_point = 500

    # Load odometry and point clouds
    dataset = pykitti.odometry(basedir, sequence, frames=range(0, max_range, 1))

    # Plot this
    f2 = plt.figure()
    ax2 = f2.add_subplot(111, projection='3d')

    draw_point_cloud(dataset, pcs_step, take_each_n_point, ax2, max_range)
    trajectory = get_trajectory(dataset)
    trajectory = to_normal_axes(trajectory)

    ax2.scatter(trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                c='red')

    ax2.set_zlim(-60, 60)
    ax2.set_ylim(-50, 50)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    plt.show()

    # TODO:: features
    exit()
    voxel_size = 0.5  # means 50cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result_part(source_down, target_down,
                                  result_ransac.transformation)

    result_icp = refine_registration(source, target,
                                     source_fpfh, target_fpfh, voxel_size)
    print(result_icp)
    draw_registration_result_part(source, target, result_icp.transformation)

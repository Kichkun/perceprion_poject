#Trajectory from groundtruth

import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from mpl_toolkits.mplot3d import Axes3D

import pykitti

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


trajectory = np.zeros((0, 3))
pc_s = np.zeros((0,3))

# Collect point clouds
for i in np.arange(0, max_range, pcs_step):
    curr = dataset.poses[i][:,3]

    # Select n points from i-th point cloud
    pc = dataset.get_velo(i)
    pc = pc[range(0, pc.shape[0], take_each_n_point)]

    # Transform from velodyne (X) to left camera coordinates (x): x = Tr * X
    Tr = dataset.calib.T_cam0_velo[:3,:]
    pc = np.array([dot(Tr, i) for i in pc])
    pc += curr[:3]

    pc_s = np.vstack((pc_s, pc))

# Collect odometry
for i in np.arange(0, max_range):
    curr = dataset.poses[i][:, 3]
    trajectory = np.vstack((trajectory, curr[:3]))

# Plot this
f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')

ax2.scatter(pc_s[:, 0],
            pc_s[:, 2],
            pc_s[:, 1])

ax2.scatter(trajectory[:,0],
            trajectory[:,2],
            trajectory[:,1],
            c='red')

plt.show() 









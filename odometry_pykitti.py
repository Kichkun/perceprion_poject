import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti

# Change this to the directory where you store our KITTI data
basedir = '/media/kish/f9d892c8-b4c1-46f0-95ed-56f4780088e6/dataset/dataset_sliced/' 

# Specify the dataset to load (now we have 4 sequenses with 200 binary files)
sequence = '00'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frames=range(0, 20, 5))
print(dataset.sequence_path)
# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data
second_pose = dataset.poses[1]
print(second_pose)
#first_gray = next(iter(dataset.gray))
#first_cam1 = next(iter(dataset.cam1))
#first_rgb = dataset.get_rgb(0)
#first_cam2 = dataset.get_cam2(0)
third_velo = dataset.get_velo(0)

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nSequence: ' + str(dataset.sequence))
print('\nFrame range: ' + str(dataset.frames))

print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond ground truth pose:\n' + str(second_pose))

f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range = range(0, third_velo.shape[0], 100)
ax2.scatter(third_velo[velo_range, 0],
            third_velo[velo_range, 1],
            third_velo[velo_range, 2],
            c=third_velo[velo_range, 3],
            cmap='gray')
ax2.set_title('Third Velodyne scan (subsampled)')

plt.show()

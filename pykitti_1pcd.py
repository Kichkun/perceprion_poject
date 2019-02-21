import glob
import os
from pykitti import utils
import pykitti
import open3d
from pclpy import pcl
#pykitti.raw.velo_files = "F:/velo/training/velodyne/"



def load_velo():
    """Load velodyne [x,y,z,reflectance] scan data from binary files."""
    # Find all the Velodyne files
    velo_path = "F:/velo/training/velodyne/"
    velo_files = sorted(glob.glob(velo_path))

    # Subselect the chosen range of frames, if any
    velo_files = [velo_files[i] for i in range(10)]

    print('Found ' + str(len(velo_files)) + ' Velodyne scans...')

    # Read the Velodyne scans. Each point is [x,y,z,reflectance]
    velo = utils.load_velo_scans(velo_files)

    print('done.')
load_velo()
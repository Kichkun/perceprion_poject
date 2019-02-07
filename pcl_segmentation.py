import math
import os

import numpy
import numpy as np
import pytest

import pclpy
from pclpy import pcl


def test_data(*args):
    return os.path.join("test_data", *args)


def make_pt(x, y, z):
    pt = pcl.point_types.PointXYZRGBA()
    pt.x = x
    pt.y = y
    pt.z = z
    return pt


def test_region_growing(path):
    pc = pcl.PointCloud.PointXYZ()
    reader = pcl.io.PCDReader()
    reader.read(path, pc)
    print(pc.points)
    rg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()
    rg.setInputCloud(pc)
    normals_estimation = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    normals_estimation.setInputCloud(pc)
    normals = pcl.PointCloud.Normal()
    normals_estimation.setRadiusSearch(0.35)
    normals_estimation.compute(normals)
    rg.setInputNormals(normals)

  #  rg.setMaxClusterSize(1000000)
   # rg.setMinClusterSize(10)
   # rg.setNumberOfNeighbours(5)
    #rg.setSmoothnessThreshold(5 / 180 * math.pi)
   # rg.setCurvatureThreshold(5)
   # rg.setResidualThreshold(5)
    clusters = pcl.vectors.PointIndices()
    rg.extract(clusters)


    pc = rg.getColoredCloud()


    viewer = pcl.visualization.PCLVisualizer("viewer")
    viewer.setBackgroundColor(0, 0, 0)
    rgb = pcl.visualization.PointCloudGeometryHandlerXYZ.PointXYZRGB(pc)
    viewer.addPointCloud(pc, rgb, "sample cloud")
    viewer.setPointCloudRenderingProperties(0, 3, "sample cloud")
    viewer.addCoordinateSystem(1.0)
    viewer.resetCamera()
    while not viewer.wasStopped():
        viewer.spinOnce(10)
    #assert max([len(c.indices) for c in clusters]) == 2449  # ground

for i in range(0,9):
    test_region_growing("dataset/pcds_00/00000"+str(i)+".pcd")
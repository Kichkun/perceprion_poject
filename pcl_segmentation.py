import math
import os
from matplotlib import pyplot as plt
import numpy
import numpy as np
import pytest
from collections import Counter
import pclpy
from pclpy import pcl
from filter_ground_find_segments import params_to_return

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
    pcd = pcl.PointCloud.PointXYZ()
    reader = pcl.io.PCDReader()
    reader.read(path, pc)
    print(len(pc.points))
    a = []

    for i in range(0, len(pc.points)):
        a.append(pc.points[i].z)
    #(plt.hist(a))
   # plt.show()
    a = np.array(a)
    #trashold =1
    minsc = a.mean()-10 #BEST FOR NOT TURNED
    maxsc = a.mean()-0.5#BEST FOR NOT TURNED
    #minsc = a.mean() -10  # BEST FOR TURNED
    #maxsc = a.mean()  # BEST FOR  TURNED

    sc = 0

    for i in range(0,  len(pc.points)):
       if (pc.points[i].z <minsc)or(pc.points[i].z >maxsc):
           sc +=1
           pcd.points.append(pc.points[i])
    print(sc)
   #rg =pcl.segmentation.OrganizedMultiPlaneSegmentation.PointXYZ_Normal()
    rg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()
    rg.setInputCloud(pcd)
    normals_estimation = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    normals_estimation.setInputCloud(pcd)
    normals = pcl.PointCloud.Normal()
    normals_estimation.setRadiusSearch(0.35)
    normals_estimation.compute(normals)
    rg.setInputNormals(normals)

    rg.setMaxClusterSize(10000)
    rg.setMinClusterSize(10)
    #rg.setNumberOfNeighbours(5)
    rg.setSmoothnessThreshold(5 / 180 * math.pi)
    rg.setCurvatureThreshold(15)
    rg.setResidualThreshold(10)#filter more ground
   # indices = pc.extract_clusters(tolerance=0.5, min_size=10, max_size=100000) #euclidean 1 arrow

    clusters = pcl.vectors.PointIndices()
    #rg.extract(indices) # euclidean
    rg.extract(clusters)

    pcd = rg.getColoredCloud()


    viewer = pcl.visualization.PCLVisualizer("viewer")
    viewer.setBackgroundColor(0, 0, 0)
    rgb = pcl.visualization.PointCloudGeometryHandlerXYZ.PointXYZRGB(pcd)
    viewer.addPointCloud(pcd, rgb, "sample cloud")
    viewer.setPointCloudRenderingProperties(0, 3, "sample cloud")
    viewer.addCoordinateSystem(1.0)
    viewer.resetCamera()
    while not viewer.wasStopped():
        viewer.spinOnce(10)
    #assert max([len(c.indices) for c in clusters]) == 2449  # ground
for i in range(0,9):
    test_region_growing("dataset/pcds_00/00009"+str(i)+".pcd")
    #test_region_growing("dataset/turned_pcds/" + str(i) + "_new.pcd")
    #повернутые облака(в реальных координатах поместить сюда! совместить облака в сотню, протестить с фильтром земли и без
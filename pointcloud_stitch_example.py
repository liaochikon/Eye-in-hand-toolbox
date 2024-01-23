from RealSense import realsense
from TMRobot import TMRobot
import numpy as np
import open3d as o3d

pic1 = [ 191.6897,   -333.4934,    650.9208,    142.2055,     -3.365571,  122.1952  ]
pic2 = [274.2222,   208.3673,   586.8492,   132.9971,    -4.144691,  58.79302 ]

workspace = [ 1, 0.4, 0.5, -0.2, 1, -0.2]
object_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(workspace[1], workspace[3], workspace[5]), max_bound=(workspace[0], workspace[2], workspace[4]))

win_ip = "192.168.0.168"
wsl_ip = "172.18.42.135"
tcp_port = 6680
move_port = 8000
tm_robot = TMRobot.TM5_900(win_ip, wsl_ip, tcp_port, move_port)

def Offset_T(t_offset = [0, 0, 0]):
    offset_T = np.array([[-9.99667144e-01,  2.54538765e-02, -4.20727984e-03,  3.63577164e+01],
                         [-2.53933752e-02, -9.99581490e-01, -1.38571629e-02, -8.71275695e+01],
                         [-4.55823756e-03, -1.37457134e-02,  9.99895133e-01, -2.69572034e+01],
                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    offset_T[0][3] = offset_T[0][3] + t_offset[0] - 25
    offset_T[1][3] = offset_T[1][3] + t_offset[1] - 60
    offset_T[2][3] = offset_T[2][3] + t_offset[2] - 10
    offset_T[:3, 3] = offset_T[:3, 3] / 1000
    return np.linalg.inv(offset_T)

def Get_PC():
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(realsense.Get_PointCloud(is_temporal_filter = False, sample_length=50))
    T_tcp = tm_robot.Get_TCP_T()
    T_tcp[:3, 3] = T_tcp[:3, 3] / 1000
    T = np.matmul(T_tcp, Offset_T())
    pc.transform(T)
    return pc.crop(object_bbox)

def test():
    tm_robot.Go_to(pic1)
    T_tcp_1 = tm_robot.Get_TCP_T()
    T_tcp_1[:3, 3] = T_tcp_1[:3, 3] / 1000
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(realsense.Get_PointCloud(is_temporal_filter = False, sample_length=50))
    pc1 = pc1.crop(object_bbox)

    tm_robot.Go_to(pic2)
    T_tcp_2 = tm_robot.Get_TCP_T()
    T_tcp_2[:3, 3] = T_tcp_2[:3, 3] / 1000
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(realsense.Get_PointCloud(is_temporal_filter = False, sample_length=50))
    pc2 = pc2.crop(object_bbox)

    step = -5
    for i in range(50):
        t_offset = [0, 0, step * i]
        print(t_offset)

        T_pc1 = np.matmul(T_tcp_1, Offset_T(t_offset))
        pc1_test = o3d.geometry.PointCloud()
        pc1_test.points = pc1.points
        pc1_test.transform(T_pc1)
        pc1_test.paint_uniform_color([0.5, 0, 0])
        
        T_pc2 = np.matmul(T_tcp_2, Offset_T(t_offset))
        pc2_test = o3d.geometry.PointCloud()
        pc2_test.points = pc2.points
        pc2_test.transform(T_pc2)
        pc1_test.paint_uniform_color([0, 0.5, 0])
        o3d.visualization.draw_geometries([pc1_test, pc2_test, Origin])


Origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
Origin.scale(0.2, center=(0, 0, 0))

tm_robot.Go_to(pic1)
pc1 = Get_PC()
pc1.paint_uniform_color([0.5, 0, 0])
tm_robot.Go_to(pic2)
pc2 = Get_PC()
pc2.paint_uniform_color([0, 0.5, 0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([*np.asfarray(pc1.points), *np.asfarray(pc2.points)])
#pcd.paint_uniform_color([0.5, 0.2, 0.2])
pcd = pcd.crop(object_bbox)

o3d.visualization.draw_geometries([pcd, Origin])
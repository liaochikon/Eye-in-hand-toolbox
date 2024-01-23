import numpy as np
from scipy.spatial.transform import Rotation

def Draw_Origin(R, t, ax, scale = 1):
    r0 = R[:, 0].reshape(3) * scale
    r1 = R[:, 1].reshape(3) * scale
    r2 = R[:, 2].reshape(3) * scale
    
    ax.quiver(t[0], t[1], t[2], r0[0], r0[1], r0[2], color='r')
    ax.quiver(t[0], t[1], t[2], r1[0], r1[1], r1[2], color='g')
    ax.quiver(t[0], t[1], t[2], r2[0], r2[1], r2[2], color='b')

def Draw_Camera(K, R, t, cam_text, ax, f=1):
    ax.text(t[0][0], t[1][0], t[2][0], cam_text)
    r0 = R[:, 0].reshape(3)
    r1 = R[:, 1].reshape(3)
    r2 = R[:, 2].reshape(3)
    ax.text(t[0][0], t[1][0], t[2][0], cam_text)
    ax.quiver(t[0], t[1], t[2], r0[0], r0[1], r0[2], color='r')
    ax.quiver(t[0], t[1], t[2], r1[0], r1[1], r1[2], color='g')
    ax.quiver(t[0], t[1], t[2], r2[0], r2[1], r2[2], color='b')
    vec = np.zeros(3)
    vec[0] = K[0][2] / K[0][0] * f
    vec[1] = K[1][2] / K[1][1] * f
    vec[2] = f
    t_T = t.reshape(3)
    lt = (-vec[0]) * r0 + (-vec[1]) * r1 + vec[2] * r2 + t_T
    lb = (-vec[0]) * r0 + vec[1] * r1 + vec[2] * r2 + t_T
    rt = vec[0] * r0 + (-vec[1]) * r1 + vec[2] * r2 + t_T
    rb = vec[0] * r0 + vec[1] * r1 + vec[2] * r2 + t_T
    ax.plot3D(xs=(t_T[0], lt[0]),
              ys=(t_T[1], lt[1]),
              zs=(t_T[2], lt[2]), color='k')
    ax.plot3D(xs=(t_T[0], rt[0]),
              ys=(t_T[1], rt[1]),
              zs=(t_T[2], rt[2]), color='k')
    ax.plot3D(xs=(t_T[0], lb[0]),
              ys=(t_T[1], lb[1]),
              zs=(t_T[2], lb[2]), color='k')
    ax.plot3D(xs=(t_T[0], rb[0]),
              ys=(t_T[1], rb[1]),
              zs=(t_T[2], rb[2]), color='k')

    ax.plot3D(xs=(lt[0], rt[0]),
              ys=(lt[1], rt[1]),
              zs=(lt[2], rt[2]), color='k')
    ax.plot3D(xs=(rt[0], rb[0]),
              ys=(rt[1], rb[1]),
              zs=(rt[2], rb[2]), color='k')
    ax.plot3D(xs=(rb[0], lb[0]),
              ys=(rb[1], lb[1]),
              zs=(rb[2], lb[2]), color='k')
    ax.plot3D(xs=(lb[0], lt[0]),
              ys=(lb[1], lt[1]),
              zs=(lb[2], lt[2]), color='k')

def Draw_Aruco(ax, aruco_length):
    ax.plot3D(xs=(aruco_length / 2, -aruco_length / 2),
                ys=(aruco_length / 2, aruco_length / 2),
                zs=(0, 0), color='k')
    ax.plot3D(xs=(-aruco_length / 2, -aruco_length / 2),
            ys=(aruco_length / 2, -aruco_length / 2),
            zs=(0, 0), color='k')
    ax.plot3D(xs=(-aruco_length / 2, aruco_length / 2),
            ys=(-aruco_length / 2, -aruco_length / 2),
            zs=(0, 0), color='k')
    ax.plot3D(xs=(aruco_length / 2, aruco_length / 2),
            ys=(-aruco_length / 2, aruco_length / 2),
            zs=(0, 0), color='k')

def R_and_t_to_T(R, t):
    T = np.hstack((R, t))
    T = np.vstack((T, [0, 0, 0, 1]))
    return T

def T_to_R_and_t(T):
    Rt = T[:3]
    R = Rt[:, :3]
    t = Rt[:, 3].reshape((-1, 1))
    return R, t

def Rotation_Matrix_to_Quaternion(R):
    r = Rotation.from_matrix(R)
    return r.as_quat()

def Quaternion_to_Rotation_Matrix(q):
    r = Rotation.from_quat(q)
    return r.as_matrix()

def Rotation_Matrix_to_Rotation_Vector(R):
    r = Rotation.from_matrix(R)
    return r.as_rotvec()

def Rotation_Vector_to_Rotation_Matrix(v):
    r = Rotation.from_rotvec(v)
    return r.as_matrix()

def Euler_Angle_to_Rotation_Matrix(e):
    r = Rotation.from_euler('xyz', e, degrees=True)
    return r.as_matrix()

def TM_Format_to_T(tm):
    t = tm[:3]
    e = tm[3:]
    R = Rotation.from_euler('xyz', e, degrees=True).as_matrix()
    t = np.array(t).reshape((-1, 1))
    Rt = np.hstack((R, t))
    T = np.vstack((Rt, [0, 0, 0, 1]))
    return T

def T_to_TM_Format(T):
    t = T[:3, 3]
    e = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    tm = np.hstack((t, e))
    return tm

def T_Rx_Advence(T, rx_offset):
    result_T = T
    R = result_T[:3, :3]
    R_z = R[:, 2]
    result_T[:3, 3] += R_z * rx_offset
    return result_T

def T_Ry_Advence(T, ry_offset):
    result_T = T
    R = result_T[:3, :3]
    R_z = R[:, 2]
    result_T[:3, 3] += R_z * ry_offset
    return result_T

def T_Rz_Advence(T, rz_offset):
    result_T = T.copy()
    R = result_T[:3, :3]
    R_z = R[:, 2]
    result_T[:3, 3] += R_z * rz_offset
    return result_T

def rotation_matrix(x, y, z):
    c1 = np.cos(x * np.pi / 180)
    s1 = np.sin(x * np.pi / 180)
    c2 = np.cos(y * np.pi / 180)
    s2 = np.sin(y * np.pi / 180)
    c3 = np.cos(z * np.pi / 180)
    s3 = np.sin(z * np.pi / 180)
    matrix=np.array([[c2*c3, -c2*s3, s2],
                     [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                     [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    matrix[:, 0] = matrix[:, 0] / np.linalg.norm(matrix[:, 0])
    matrix[:, 1] = matrix[:, 1] / np.linalg.norm(matrix[:, 1])
    matrix[:, 2] = matrix[:, 2] / np.linalg.norm(matrix[:, 2])
    return matrix

def Depth_to_PointCloud(K, depth_img, min_dist = 0, max_dist = 2):
    K_inv = np.linalg.inv(K)
    point_cloud = []
    #percent = 0
    for v in range(depth_img.shape[0]):
        for u in range(depth_img.shape[1]):
            s = depth_img[v][u] / 1000 #z-distance(depth) in meter
            if s <= min_dist or s >= max_dist:
                continue
            w = np.array([u, v, 1]).reshape(3, 1)
            W = s * np.dot(K_inv, w)
            point_cloud.append(W)
        percent_temp = int(v / depth_img.shape[0] * 100) + 1
        #if percent_temp != percent:
        #    percent = percent_temp
        #    print(percent, "%")
    point_cloud = np.array(point_cloud)
    return point_cloud.reshape((len(point_cloud), 3))
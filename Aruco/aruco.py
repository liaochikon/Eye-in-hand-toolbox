import numpy as np
import cv2

class ARUCO_DICT():
    def __init__(self):
        self.DICT_4X4_50 = cv2.aruco.DICT_4X4_50
        self.DICT_4X4_100 = cv2.aruco.DICT_4X4_100
        self.DICT_4X4_250 = cv2.aruco.DICT_4X4_250
        self.DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
        self.DICT_5X5_50 = cv2.aruco.DICT_5X5_50
        self.DICT_5X5_100 = cv2.aruco.DICT_5X5_100
        self.DICT_5X5_250 = cv2.aruco.DICT_5X5_250
        self.DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
        self.DICT_6X6_50 = cv2.aruco.DICT_6X6_50
        self.DICT_6X6_100 = cv2.aruco.DICT_6X6_100
        self.DICT_6X6_250 = cv2.aruco.DICT_6X6_250
        self.DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000
        self.DICT_7X7_50 = cv2.aruco.DICT_7X7_50
        self.DICT_7X7_100 = cv2.aruco.DICT_7X7_100
        self.DICT_7X7_250 = cv2.aruco.DICT_7X7_250
        self.DICT_7X7_1000 = cv2.aruco.DICT_7X7_1000
        self.DICT_ARUCO_ORIGINAL = cv2.aruco.DICT_ARUCO_ORIGINAL
        self.DICT_APRILTAG_16h5 = cv2.aruco.DICT_APRILTAG_16h5
        self.DICT_APRILTAG_25h9 = cv2.aruco.DICT_APRILTAG_25h9
        self.DICT_APRILTAG_36h10 = cv2.aruco.DICT_APRILTAG_36h10
        self.DICT_APRILTAG_36h11 = cv2.aruco.DICT_APRILTAG_36h11

def R_and_t_to_T(R, t):
    T = np.hstack((R, t))
    T = np.vstack((T, [0, 0, 0, 1]))
    return T

def T_to_R_and_t(T):
    Rt = T[:3]
    R = Rt[:, :3]
    t = Rt[:, 3].reshape((-1, 1))
    return R, t

class Aruco():
    def __init__(self, aruco_dict, id, resolution = 300, aruco_params = None):
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.id = id
        self.resolution = resolution
        self.tag = self.Spawn_Aruco()
        if aruco_params == None:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = aruco_params

    def Spawn_Aruco(self):
        tag = np.zeros((self.resolution, self.resolution, 1), dtype="uint8")
        cv2.aruco.drawMarker(self.aruco_dict, self.id, self.resolution, tag, 1)
        return tag

def Detect_Aruco(frame, K, D, aruco_length, aruco_dict, aruco_params, is_draw_aruco = False, is_millimeter = False):
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    if len(corners) == 0:
        return False, [], [], [], []
    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_length, K, D)
    T_cam_to_aruco_result = []
    T_aruco_to_cam_result = []
    id_result = []
    corner_result = []
    for c, id, t, r in zip(corners, ids, tvec, rvec):
        cv2.drawFrameAxes (frame, K, D, r, t, 0.08)
        id = id[0]
        c = c[0]
        R_cam_to_aruco = cv2.Rodrigues(r[0])[0]
        t_cam_to_aruco = t[0].reshape(-1, 1)
        if is_millimeter:
            t_cam_to_aruco *= 1000
        T_cam_to_aruco = R_and_t_to_T(R_cam_to_aruco, t_cam_to_aruco)
        T_aruco_to_cam = np.linalg.inv(T_cam_to_aruco)
        T_cam_to_aruco_result.append(T_cam_to_aruco)
        T_aruco_to_cam_result.append(T_aruco_to_cam)
        id_result.append(id)
        corner_result.append(c)
        
    if is_draw_aruco:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return True, T_cam_to_aruco_result, T_aruco_to_cam_result, id_result, corner_result

def Corners_Center(corners):
    return np.mean(corners, axis=0)
        

        
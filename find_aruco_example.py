from RealSense import realsense
from Aruco import aruco
from Robot_Util import util
import numpy as np
import matplotlib.pyplot as plt
import cv2

aruco_5x5_100_id_24 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 24, 300)
aruco_length = 0.08

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
K = realsense.Get_Color_K()
D = np.array([0.0,0.0,0.0,0.0,0.0,])

Workspace = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

while True:
    plt.cla()
    ret, frame = cap.read()
    ret, T_cam_to_aruco_result, T_aruco_to_cam_result, id_result = aruco.Detect_Aruco(frame, K, D, aruco_length, aruco_5x5_100_id_24.aruco_dict, aruco_5x5_100_id_24.aruco_params, True)
    if ret:
        for id, T in zip(id_result, T_aruco_to_cam_result):
            R_aruco_to_cam, t_aruco_to_cam = util.T_to_R_and_t(T)
            cam_text = "X : {:.3f}, Y : {:.3f}, Z : {:.3f}".format(t_aruco_to_cam[0][0], t_aruco_to_cam[1][0], t_aruco_to_cam[2][0])
            if id == aruco_5x5_100_id_24.id:
                util.Draw_Camera(K, R_aruco_to_cam, t_aruco_to_cam, cam_text, ax, f=0.08)
    
    util.Draw_Aruco(ax, aruco_length)
    ax.set_xlim3d(Workspace[0], Workspace[1])
    ax.set_ylim3d(Workspace[2], Workspace[3])
    ax.set_zlim3d(Workspace[4], Workspace[5])
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    ax.set_zlabel('z')
    plt.show(block=False)
    plt.pause(0.001)
    cv2.imshow("s", frame)
    if cv2.waitKey(1) == ord('q'):
        break
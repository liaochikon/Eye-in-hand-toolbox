from RealSense import realsense
from Robot_Util import util
from TMRobot import TMRobot
from Aruco import aruco
from Eye_In_Hand import bundle_adjustment
import numpy as np
import cv2
import matplotlib.pyplot as plt

win_ip = "192.168.0.212"
wsl_ip = "172.21.242.201"
tcp_port = 6680
move_port = 8000
TMRobot = TMRobot.TM5_900(win_ip, wsl_ip, tcp_port, move_port)

tcp_tip = [813.2574,   -41.73158,   72.21378,  167.1379,     8.028033,  88.10474 ]
p1 = [537.2827, 260.1796, 451.7567, 135.7346, -2.38456, 37.27498]
p2 = [ 191.6897,   -333.4934,    650.9208,    142.2055,     -3.365571,  122.1952  ]
p3 = [494.7805, 1.543949, 505.4083, 140.5734, -3.460447, 94.21851]
p4 = [274.2222,   208.3673,   586.8492,   132.9971,    -4.144691,  58.79302 ]
p5 = [591.7986, -187.6871, 437.4228, 144.9929, 0.9061133, 126.4476]
p6 = [ 718.8137,  -227.4249,   409.8435,   164.8534,   -28.48721,   90.77137]
p7 = [7.765230e+02,  4.989940e-01,  4.817713e+02,  1.680848e+02, -1.459568e-02, 9.638807e+01]
p8 = [692.5287,  209.8485,  400.0923,  171.7128,   26.444,    89.97701]
points = [p1 ,p2 ,p3 ,p4 ,p5 ,p6 ,p7 ,p8]

T_aruco_to_tip = np.eye(4)
T_aruco_to_tip[:3, 3] = np.array([45, 45, 50])
T_base_to_tip = util.TM_Format_to_T(tcp_tip)
T_tip_to_aruco = np.linalg.inv(T_aruco_to_tip)

aruco_5x5_100_id_24 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 24, 300)
aruco_length = 0.08

K = realsense.Get_Color_K()
D = np.array([0.0,0.0,0.0,0.0,0.0,])

#Workspace = [-1, 1, -1, 1, -1, 1]
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

# init params
T_cam_to_aruco_list = []
T_base_to_tcp_list = []
W_list = []
w_list = []
real_ans_list = []
W_real_ans_list = []

#W_real_ans_24 = np.array([778.1096,    73.3577,    21.31925]).reshape((-1, 1))
W_real_ans_1 = np.array([850.5286,     22.77639,    23.24156]).reshape((-1, 1))

R_init = util.Euler_Angle_to_Rotation_Matrix([0, 0, 180])
EH = bundle_adjustment.Eye_In_Hand_Camera(K, R_init, np.zeros((3, 1)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for p in points:
    TMRobot.Go_to(p)
    frame = realsense.Get_RGB_Frame()
    ret, T_cam_to_aruco_result, T_aruco_to_cam_result, id_result, corner_result = aruco.Detect_Aruco(frame, 
                                                                                                    K, 
                                                                                                    D, 
                                                                                                    aruco_length, 
                                                                                                    aruco_5x5_100_id_24.aruco_dict, 
                                                                                                    aruco_5x5_100_id_24.aruco_params, 
                                                                                                    True, 
                                                                                                    True)
    if ret == False:
        continue
    for id, T, c in zip(id_result, T_cam_to_aruco_result, corner_result):
        
        T_base_to_tcp = TMRobot.Get_TCP_T()
        T_cam_to_aruco_list.append(T)
        T_base_to_tcp_list.append(T_base_to_tcp)
        
        T_W = np.matmul(T_base_to_tcp, EH.T_tcp_to_cam())
        T_W = np.matmul(T_W, T)
        W = T_W[:3, 3].reshape((-1, 1))
        W_list.append(W)
        w = aruco.Corners_Center(c)
        w_list.append(w)
        cv2.imshow("pic", frame)
        cv2.waitKey(1)
        #if id == 24:
        #    real_ans_list.append(W_real_ans_24)
        if id == 1:
            real_ans_list.append(W_real_ans_1)
        print("get pic!!")
        print(id)

EH.Eye_Hand_Calibration(W_list, w_list, T_cam_to_aruco_list, T_base_to_tcp_list, real_ans_list)

T = EH.T_tcp_to_cam()

Workspace = [-1, 1, -1, 1, -1, 1]

R, t = util.T_to_R_and_t(T)
util.Draw_Origin(np.eye(3), np.zeros((3, 1)), ax, 100)
util.Draw_Camera(K, R, t, "", ax, 100)
ax.set_xlabel('x')
ax.set_ylabel('y') 
ax.set_zlabel('z')
plt.show()
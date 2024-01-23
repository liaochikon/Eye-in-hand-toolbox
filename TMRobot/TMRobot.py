import numpy as np
import socket, pickle
from scipy.spatial.transform import Rotation

class TM5_900():
    def __init__(self, win_ip, wsl_ip, tcp_port, move_port, tcp_T_camera = None):
        self.win_ip = win_ip
        self.wsl_ip = wsl_ip
        self.tcp_port = tcp_port
        self.move_port = move_port
        self.tcp_soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
        self.tcp_soc.bind((win_ip, tcp_port))
        self.move_soc = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
        self.move_soc.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
        self.TCP_T_Camera = tcp_T_camera

    def Get_TCP_T(self):
        a = pickle.loads(self.tcp_soc.recvfrom(1000000)[0])
        R = Rotation.from_euler('xyz', [a[3], a[4], a[5]], degrees=True).as_matrix()
        t = a[:3].reshape((-1, 1))
        Rt = np.hstack((R, t))
        T = np.vstack((Rt, [0, 0, 0, 1]))
        return T
    
    def Go_to(self, point, tol_dist = 0.02):
        Hand_T_data = pickle.dumps(point)
        self.move_soc.sendto((Hand_T_data), (self.wsl_ip, self.move_port))
        dist = 1000
        target_t = point[:3]
        while(dist > tol_dist):
            current_hand_T = self.Get_TCP_T()
            current_t = current_hand_T[:3, 3]
            dist = np.linalg.norm(current_t - target_t)
            print(dist)

    def Get_TCP_T_Camera(self, t_offset = [0, 0, 0]):
        if self.TCP_T_Camera == None:
            return []
        tcp_T_camera = self.TCP_T_Camera
        tcp_T_camera[0][3] += t_offset[0]
        tcp_T_camera[1][3] += t_offset[1]
        tcp_T_camera[2][3] += t_offset[2]
        return np.linalg.inv(tcp_T_camera)
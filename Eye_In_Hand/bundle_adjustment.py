import numpy as np
import util
from scipy.optimize import least_squares

class Eye_In_Hand_Camera():
    def __init__(self, K, R, t):
        self.K = K
        self.R_tcp_to_cam = R
        self.t_tcp_to_cam = t

    def T_tcp_to_cam(self):
        return util.R_and_t_to_T(self.R_tcp_to_cam, self.t_tcp_to_cam)
    
    def EHC_x0(self, W_list):
        q = util.Rotation_Matrix_to_Quaternion(self.R_tcp_to_cam)
        t = self.t_tcp_to_cam.ravel()
        Ws = np.array(W_list).ravel()
        x0 = np.array([*q, *t, *Ws])
        return x0
    
    def EHC_Error(self, params, w_list, T_cam_to_aruco_list, T_base_to_tcp_list, real_ans_list):
        r_index = 4
        w_index = 7
        q = params[:r_index]
        t = params[r_index:w_index].reshape((-1, 1))
        W_list = params[w_index:]
        W_list = W_list.reshape((len(W_list) // 3, 3))
        self.R_tcp_to_cam = util.Quaternion_to_Rotation_Matrix(q)
        self.t_tcp_to_cam = t

        e_list = []
        for W_real_ans, w, T_ca, T_bt in zip(real_ans_list, w_list, T_cam_to_aruco_list, T_base_to_tcp_list):
            T_cam = np.matmul(T_bt, self.T_tcp_to_cam())
            R_cam, t_cam= util.T_to_R_and_t(T_cam)
            #T_W = np.matmul(T_ca, T_cam)
            #W = T_W[:3, 3].reshape((-1, 1))

            e = Reprojection_Error(W_real_ans, self.K, R_cam, t_cam, w)

            e_list.append(e[0][0])
            e_list.append(e[1][0])
        #print(e)
        return e_list
    
    def Eye_Hand_Calibration(self, W_list, w_list, T_cam_to_aruco_list, T_base_to_tcp_list, real_ans_list):
        x0 = self.EHC_x0(W_list)
        e0 = self.EHC_Error(x0, w_list, T_cam_to_aruco_list, T_base_to_tcp_list, real_ans_list)
        res = least_squares(self.EHC_Error, x0, verbose=2, method='trf', ftol=1e-32,
                            args=(w_list, T_cam_to_aruco_list, T_base_to_tcp_list, real_ans_list))
        e = self.EHC_Error(res['x'], w_list, T_cam_to_aruco_list, T_base_to_tcp_list, real_ans_list)

        print("Before")
        print(np.sum(np.abs(e0)))
        print("After")
        print(np.sum(np.abs(e)))
        print(self.T_tcp_to_cam())

def Reprojection_Error(W, K, R, t, w):
    W = W.reshape((-1, 1))
    R_inv = np.linalg.inv(R)
    W_T = np.matmul(R_inv, W - t)
    sw = np.matmul(K, W_T)
    s = sw[2]
    w_re = sw / s
    w_re = w_re.reshape(3)[:2]
    e = w - w_re
    return e.reshape((-1, 1))
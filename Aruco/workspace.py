from Aruco import aruco
import numpy as np

aruco_length = 0.053
aruco_5x5_100_1 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 1)
aruco_5x5_100_2 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 2)
aruco_5x5_100_3 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 3)
aruco_5x5_100_4 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 4)

board_width = 321
board_height = 234.5
board_T_tag1 = np.array([[ 1,  0,  0,   board_width / 2],
                         [ 0,  1,  0, -board_height / 2],
                         [ 0,  0,  1,                 0],
                         [ 0,  0,  0,                 1]])
board_T_tag2 = np.array([[ 1,  0,  0,  -board_width / 2],
                         [ 0,  1,  0, -board_height / 2],
                         [ 0,  0,  1,                 0],
                         [ 0,  0,  0,                 1]])
board_T_tag3 = np.array([[ 1,  0,  0,   board_width / 2],
                         [ 0,  1,  0,  board_height / 2],
                         [ 0,  0,  1,                 0],
                         [ 0,  0,  0,                 1]])
board_T_tag4 = np.array([[ 1,  0,  0,  -board_width / 2],
                         [ 0,  1,  0,  board_height / 2],
                         [ 0,  0,  1,                 0],
                         [ 0,  0,  0,                 1]])

def Aruco_Workspace(frame, base_T_tcp, tcp_T_cam, K, D):
    ret, T_cam_to_aruco_result, T_aruco_to_cam_result, id_result, corner_result = aruco.Detect_Aruco(
                                                                                            frame, K, D, aruco_length, 
                                                                                            aruco_5x5_100_1.aruco_dict, 
                                                                                            aruco_5x5_100_1.aruco_params, True)
    for id, cam_T_tag in zip(id_result, T_cam_to_aruco_result):
        cam_T_tag[:3, 3] *= 1000
        board_T_tag = np.eye(4)
        if id == aruco_5x5_100_1.id:
            board_T_tag = board_T_tag1
            print("Get aruco 1")
        if id == aruco_5x5_100_2.id:
            board_T_tag = board_T_tag2
            print("Get aruco 2")
        if id == aruco_5x5_100_3.id:
            board_T_tag = board_T_tag3
            print("Get aruco 3")
        if id == aruco_5x5_100_4.id:
            board_T_tag = board_T_tag4
            print("Get aruco 4")

        base_T_cam = np.matmul(base_T_tcp, tcp_T_cam)
        base_T_tag = np.matmul(base_T_cam, cam_T_tag)
        tag_T_board = np.linalg.inv(board_T_tag)
        base_T_board = np.matmul(base_T_tag, tag_T_board)
        return base_T_board
    
def Get_Base_T_BoardPose(base_T_Board):
    tcp_T_cam = np.array([[-1,  0,  0,   0],
                          [ 0,  1,  0,   0],
                          [ 0,  0, -1, 400],
                          [ 0,  0,  0,   1]])
    base_T_Boardpose = np.matmul(base_T_Board, tcp_T_cam)
    return base_T_Boardpose

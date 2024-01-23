#!/usr/bin/env python3.8

import rospy
from tm_msgs.msg import *
from tm_msgs.srv import *
import ast
import numpy as np
import socket
import pickle

Win_ip = "172.20.10.8"
Coord_Base_Tool_port = 6680
Coord_Base_Tool_soc = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
Coord_Base_Tool_soc.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
def callback(data):
    Coord_Base_Tool = data.content
    Coord_Base_Tool = Coord_Base_Tool.split('=')[1]
    Coord_Base_Tool = Coord_Base_Tool.replace("{", "")
    Coord_Base_Tool = Coord_Base_Tool.replace("}", "")
    Coord_Base_Tool = Coord_Base_Tool.split(',')
    Coord_Base_Tool = np.asfarray(Coord_Base_Tool)
    print(Coord_Base_Tool)
    Coord_Base_Tool_data = pickle.dumps(Coord_Base_Tool)
    Coord_Base_Tool_soc.sendto((Coord_Base_Tool_data), (Win_ip, Coord_Base_Tool_port))


def ask_item_demo():
    rospy.init_node('ask_item_demo')
    rospy.Subscriber('tm_driver/svr_response', SvrResponse, callback)
    rospy.wait_for_service('tm_driver/ask_item')
    ask_item = rospy.ServiceProxy('tm_driver/ask_item', AskItem)
    
    while not rospy.is_shutdown():
        res0 = ask_item('he0', 'Coord_Base_Tool', 0)
        rospy.sleep(0.5)
    Coord_Base_Tool_soc.close()

if __name__ == '__main__':
    try:
        ask_item_demo()
    except rospy.ROSInterruptException:
        pass

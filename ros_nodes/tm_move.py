#!/usr/bin/env python3.8

import rospy
from tm_msgs.msg import *
from tm_msgs.srv import *
import numpy as np
import socket
import pickle

home = [711.8223, -89.60531, 540.267, 153.3777, 2.413842, 92.6958]
place = [ -72.9598, 648.877, 368.6339, 179.962, -1.878298 -179.3747]
workspace = [ 0.75, 0.35, 0.16, -0.21, 0.46, 0.06]

WSL_ip = "172.21.242.201"
Hand_T_port = 8000

Hand_T_soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
Hand_T_soc.bind((WSL_ip, Hand_T_port))

def callback(msg):
    rospy.loginfo(rospy.get_caller_id() + ': %s', msg.subdata)
    if msg.subcmd == '01':
        data = msg.subdata.split(',')
        if data[1] == 'true':
            rospy.loginfo('point (Tag %s) is reached', data[0])

def queue_tag_demo(by_polling):
    rospy.init_node('queue_tag_demo')

    if not by_polling:
        # listen to 'tm_driver/sta_response' topic
        rospy.Subscriber('tm_driver/sta_response', StaResponse, callback)

    # using services
    rospy.wait_for_service('tm_driver/set_event')
    rospy.wait_for_service('tm_driver/set_positions')
    rospy.wait_for_service('tm_driver/ask_sta')

    set_event = rospy.ServiceProxy('tm_driver/set_event', SetEvent)
    send_script = rospy.ServiceProxy('tm_driver/send_script', SendScript)
    ask_sta = rospy.ServiceProxy('tm_driver/ask_sta', AskSta)

    # 4 points (joint angle[rad])
    points = [
        "PTP(\"CPP\",700,-152,509,155.83,-1.13,44.58,35,200,0,false)",
        "PTP(\"CPP\",-72.9598,648.877,368.6339,179.962,-1.878298-179.3747,35,200,0,false)",
        "PTP(\"CPP\",590,-152,509,155.83,-1.13,44.58,35,200,0,false)",
        "PTP(\"CPP\",-72.9598,648.877,368.6339,179.962,-1.878298-179.3747,35,200,0,false)",
    ]

    for i in range(1):
        ok = send_script("demo" , points[i])
        print(points[i], ok)
        #set_event(SetEventRequest.TAG, i + 1, 0)

    if by_polling:
        i = 0
        while i < 4:
            rospy.sleep(0.2)
            res = ask_sta('01', str(i + 1), 1)
            if res.subcmd == '01':
                data = res.subdata.split(',')
                if data[1] == 'true':
                    rospy.loginfo('point %d (Tag %s) is reached', i + 1, data[0])
                    i = i + 1
    else:
        rospy.spin()

def get_str(a):
    s = "PTP(\"CPP\","
    for aa in a:
        s += str(aa) + ","
    s += "35,200,0,false)"
    return s

def move():
    rospy.init_node('tm_move')
    rospy.wait_for_service('tm_driver/set_event')
    rospy.wait_for_service('tm_driver/send_script')
    rospy.wait_for_service('tm_driver/ask_sta')

    send_script = rospy.ServiceProxy('tm_driver/send_script', SendScript)

    while not rospy.is_shutdown():
        print("s")
        a = pickle.loads(Hand_T_soc.recvfrom(1024)[0])
        s = get_str(a)
        ok = send_script("demo" , s)
        print(s, ok)
        #rospy.spin()

if __name__ == '__main__':
    try:
        #queue_tag_demo(False)
        move()
    except rospy.ROSInterruptException:
        pass

import cv2
from Aruco import aruco

aruco_5x5_100 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 1, 300)

cv2.imwrite(str(aruco_5x5_100.id) + ".png", aruco_5x5_100.tag)# save aruco image
cv2.imshow("ArUCo Tag", aruco_5x5_100.tag)
cv2.waitKey(0)
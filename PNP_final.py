import cv2
from numpy import *
import numpy as np
import math
object_3d_points = np.array(([-105,-54,0],
                            [-39,-109,0],
                            [-237.5,-68,0],
                            [-175,-108,0],
                            [-121,-112,0],

                             ), dtype=np.double)
object_2d_point = np.array(([1077,970],
                            [773,996],
                            [1187,648],
                            [994,722],
                            [894,812],
                            ), dtype=np.double)

camera_matrix = np.load('../mtx_LK15.npy')

dist_coefs = np.load('../dist_LK15.npy')

# 求解相机位姿
found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)
#print("rvec\n",rvec,"\ntvec\n", tvec)
rotM = cv2.Rodrigues(rvec)[0]
# np.save("rotM_LK15",rotM)
# np.save("tvec_LK15",tvec)

# 计算参数SD
temmat = mat(rotM).I * mat(camera_matrix).I * mat([1166,688,1]).T
temmat2 = mat(rotM).I * mat(tvec)
#print(temmat, temmat2)
s = temmat2[2]
s = s/temmat[2]
#print("s",s[0, 0])


# 计算世界坐标
wcpoint = mat(rotM).I * (s[0, 0] * mat(camera_matrix).I * mat([1166,688,1]).T - mat(tvec))
print("world point:\n",wcpoint)

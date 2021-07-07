import cv2
import numpy as np

mtx=np.load('mtx.npy')
dist=np.load('dist.npy')

cap = cv2.VideoCapture(0)# 调整参数实现读取视频或调用摄像头
while 1:
    ret, frame = cap.read()
    dst = cv2.undistort(frame, mtx, dist, None, mtx)
    cv2.imshow("cap", dst)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
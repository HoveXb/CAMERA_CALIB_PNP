# encoding: utf-8
# by HoveXb
# 2021-06-12
####
import cv2
import numpy as np
import glob
import argparse


def calib(opt):
    # 导入配置信息
    width, height, square_size = opt.width, opt.height, opt.square_size
    imgs_source = opt.imgs_source
    images_format = opt.image_format
    imgs_source = imgs_source + f'*.{images_format}'
    drawChessboardCorners = opt.drawChessboardCorners

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((width * height, 3), np.float32)
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp[:, :2] = (np.mgrid[0:height, 0:width] * square_size).T.reshape(-1, 2)

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob(imgs_source)

    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (height, width), None)

        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            # 是否画出角点并保存
            if drawChessboardCorners:
                cv2.drawChessboardCorners(img, (height, width), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
                i += 1
                cv2.imwrite('conimg' + str(i) + f'.{images_format}', img)
                cv2.waitKey(150)
        else:
            print("can't find cornet!")

    print("there are total ", len(img_points), "images.\n")
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    # 保存数据
    mtx_save = np.array(mtx)
    dist_save = np.array(dist)
    np.save("mtx_LK15.npy", mtx_save)
    np.save("dist_LK15.npy", dist_save)

    undistort_img = opt.undistort
    if undistort_img:
        img = cv2.imread(images[0])
        h, w = img.shape[:2]

        # If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels.
        # So it may even remove some pixels at image corners.
        # If alpha=1, all pixels are retained with some extra black images.
        # This function also returns an image ROI which can be used to crop the result.
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）

        print("------------------使用undistort函数(newcameramtx)-------------------")
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        dst1 = dst[y:y + h, x:x + w]
        cv2.imwrite('calibresult1' + f'.{images_format}', dst1)
        print("Using cv.undistort_img(),dst的大小为:", dst1.shape)

        print("------------------使用remapping-------------------")
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        dst2 = dst[y:y + h, x:x + w]
        print("Using remapping", dst2.shape)
        cv2.imwrite('calibresult2' + f'.{images_format}', dst2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs-source', type=str, default='./LK_R_pic/', help='imgs source')
    parser.add_argument('--image-format', type=str, default='png', help='imgs source')
    parser.add_argument('--width', type=int, default=8, help='how many inner squares there are in the chessboard')
    parser.add_argument('--height', type=int, default=6, help='how many inner squares there are in the chessboard')
    parser.add_argument('--square-size', type=int, default=1, help='square size of chess board(mm)')
    parser.add_argument('--drawChessboardCorners', action='store_true', help='drawChessboardCorners or not')
    parser.add_argument('--undistort', action='store_true', help='undistort')
    opt = parser.parse_args()
    calib(opt)

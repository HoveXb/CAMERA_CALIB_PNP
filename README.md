# CAMERA_CALIB_PNP
>该仓库包含相机内参标定代码，以及PNP转换的旋转平移矩阵代码
## 1.相机内参标定
1. 用待标定相机拍摄不少于10张，各个角度的标定版照片，并将这些照片置于一个文件夹中
2. 运行`python calib.py --imgs-source img_path --image-format img_format --width width --heifht height`
* 上述代码中，`img_path`为标定图片文件夹路径，`img_format`为图像格式，如：jpg，png。`width,height`为标定板内部角点在长宽方向的数量


## 2.PNP转换矩阵
1. 利用测量工具测得标定版上的在实际三维坐标系下的坐标，并用对应要使用PNP算法的相机拍摄下照片
2. 利用ps或其他工具找到标定版上的点在图像坐标系下的坐标(横为x，纵为y)
3. 将各对应坐标输入`PNP_final.py`中，得到转换矩阵和平移矩阵
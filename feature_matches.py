import cv2
import numpy as np

# 读取图像
image1 = cv2.imread('./data/owl.jpg', 0)  # 灰度模式读取
image2 = cv2.imread('./data/test4.png', 0)

##################
# Use SIFT
##################
# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测特征点并计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 创建BFMatcher对象, cv2.NORM_L2：适用于SIFT、SURF等浮点描述符，它使用欧几里得距离（L2范数）来计算匹配。
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


##################
# Use ORB
##################
# # 初始化ORB检测器
# orb = cv2.ORB_create()

# # 检测特征点并计算描述符
# keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# # 创建匹配器, cv2.NORM_HAMMING：适用于ORB、BRIEF、BRISK等二进制描述符，它使用汉明距离来计算匹配。汉明距离是两个等长字符串之间对应位置的不同字符的个数，适用于处理二进制字符串的比较。
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


##################
# Start matching
##################
# 匹配描述符 - 进行匹配
matches = bf.match(descriptors1, descriptors2)

# 按距离排序
matches = sorted(matches, key=lambda x:x.distance)

# 绘制前N个匹配项
N = 50  # 可以根据需要调整这个值
matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:N], None, flags=2)

# 显示图像
cv2.imshow('Feature Matches', matched_img)


# 获取匹配关键点的位置
matched_points1 = [keypoints1[m.queryIdx].pt for m in matches[:N]]  # 图像1中的匹配关键点位置
matched_points2 = [keypoints2[m.trainIdx].pt for m in matches[:N]]  # 图像2中的匹配关键点位置

# 如果需要，可以转换成整数坐标
matched_points1 = [tuple(map(int, pt)) for pt in matched_points1]
matched_points2 = [tuple(map(int, pt)) for pt in matched_points2]

# 打印或处理匹配关键点的位置
for pt1, pt2 in zip(matched_points1, matched_points2):
    print(f"Image 1 point: {pt1} is matched to Image 2 point: {pt2}")

# 将匹配点转换为float32类型的numpy数组
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 使用RANSAC算法计算透视变换矩阵 H
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# 应用透视变换将原图映射到新图上
height, width = image2.shape[:2] # img灰度读取是2维, 彩色读取是3维
im1Reg = cv2.warpPerspective(image1, H, (width, height)) # 使用 cv2.warpPerspective() 应用透视变换后，im1Reg 变量将包含变换后的图像1（image1），其中只有映射（或变换）的部分有图像的数值，而其他地方的数值通常为0（黑色）

# 创建一个与图像2大小相同的空白图像
overlay = np.zeros_like(image2)

# 将变换后的图像1复制到overlay中的对应位置
# 这里假设im1Reg与image2的尺寸相同, 虽然height, width一定不变, 但如果img1是单通道, img2是三通道就会出现size不同, 好在这里img1和img2都采用灰度图, i.e.单通道
overlay = np.where(im1Reg > 0, im1Reg, overlay)

# 将映射后的图像1以一定透明度叠加在图像2上
# alpha 表示图像1的透明度，beta 表示图像2的透明度, 同一像素点的透明度只和如果>1, 那亮度会更高, 如果要保持亮度, 需要同一像素点的透明度sum=1
# gamma 是一个加到最终结果上的标量值
alpha = 0.5
beta = 1
gamma = 0

# 进行加权叠加
blend = cv2.addWeighted(image2, beta, overlay, alpha, gamma)

# 显示映射后的图像
cv2.imshow('Registered Image', blend)
cv2.waitKey(0)
cv2.destroyAllWindows()



##################
# Effect comparision
##################
# 效果比较：
# 准确性：在图像匹配和识别的准确性方面，SIFT通常被认为是更加准确的，因为其描述符更加详细，能够捕获更多信息。
# 速度：ORB在速度上占优，特别是在对实时性要求较高的应用中。
# 鲁棒性：SIFT对尺度和旋转的鲁棒性更好，尤其是在面对尺度变化较大的图像时。
# 资源消耗：SIFT在计算和内存资源消耗上比ORB大，特别是在处理高分辨率图像时。


##################
# Attribute explanation
##################
# 关键点（Keypoints）: 关键点是图像中的特定位置，它们是在不同的图像之间可靠匹配的独特点。它们通常是图像中明显的角点、边缘或斑点。在SIFT算法中，关键点不仅包含它们的位置信息，还包含了尺度和方向信息，这使得SIFT描述符对旋转和尺度变化保持不变性。
# 描述符（Descriptors）: 描述符是对关键点周围像素的强度分布的一种量化表示，它提供了关键点邻域内的图像内容信息。在SIFT算法中，描述符是一个128维的向量，它编码了关键点附近区域的局部梯度方向和大小信息。

# 灰度图像：当你以灰度模式读取图像时，使用的是 cv2.imread('filename', 0) 或者 cv2.imread('filename', cv2.IMREAD_GRAYSCALE)。这会读取一个二维数组，其中的两个维度分别代表图像的高度和宽度。像素值直接代表灰度级别，通常范围从0（黑色）到255（白色）。因为灰度值本身就包含在每个像素的单一数值中，所以不需要第三个维度。
# 彩色图像：当你以默认方式（彩色）读取图像时，使用的是 cv2.imread('filename') 或者 cv2.imread('filename', cv2.IMREAD_COLOR)。这会读取一个三维数组，其中前两个维度仍然代表图像的高度和宽度，第三个维度代表颜色通道。对于标准的RGB图像，有三个颜色通道（红、绿、蓝）。在OpenCV中，默认的颜色顺序实际上是BGR，而不是人们通常熟悉的RGB顺序。

# 在OpenCV函数 cv2.findHomography() 中：
# H 是计算出的透视变换矩阵（Homography Matrix）。透视变换矩阵是一个3x3的矩阵，它可以将一个平面内的点映射到另一个平面内的点。在图像匹配的上下文中，它用来转换图像的视角，以便两个图像中相对应的点可以重合。
# mask 是一个输出数组，它提供了每个点对是否是一个内点（inlier）还是外点（outlier）的信息。内点是指那些在透视变换后，两个图像中相对应的点能够很好地对齐的点。外点是指那些对齐效果不好的点，通常这些点是错误的匹配。这个数组可以用于过滤掉错误的匹配。
# 参数 cv2.RANSAC 表明用于计算Homography的算法是RANSAC（Random Sample Consensus）。RANSAC是一种强大的算法，用于从一组包含外点的数据中估计数学模型的参数。在寻找Homography时，它试图找到最佳的变换矩阵，这个矩阵使得大多数点对（内点）之间的误差最小。
# 参数 5.0 是RANSAC重新投影误差的阈值。这个值表示一个点对被认为是内点的条件，即原始点通过Homography变换后与目标点之间的距离应该小于这个阈值（以像素为单位）。如果两个点之间的距离小于这个阈值，这个点对就被认为是内点，否则就是外点。
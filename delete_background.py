import cv2
import numpy as np

def remove_border_white_background(input_path, output_path):
    # 读取图像
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:  # 如果是RGB格式，转换为RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 创建一个3通道的RGB图像用于泛洪填充
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img_rgb.shape[:2]

    # 创建一个和原图像一样大小的遮罩层
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # 泛洪填充的参数
    floodfill_flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    
    # loDiff, upDiff 说明: 如果您在进行泛洪填充来去除白色背景，并设置loDiff = (10, 10, 10)和upDiff = (10, 10, 10)，那么算法会填充那些颜色值在当前像素颜色值上下浮动10个单位内的像素。
    loDiff, upDiff = (10, 10, 10), (10, 10, 10)  # 调整这些值以更好地覆盖白色区域

    # 从四个角进行泛洪填充
    for x in [0, w - 1]:
        for y in [0, h - 1]:
            cv2.floodFill(img_rgb, mask, (x, y), 0, loDiff, upDiff, floodfill_flags) # 根据OpenCV文档，遮罩层的填充值默认为1, 但这里遮.罩层会被设置为255

    # 将非边缘部分的遮罩转换为透明
    mask = mask[1:-1, 1:-1]  # 去掉边界的遮罩层
    img[mask == 255] = [0, 0, 0, 0]

    # 保存为PNG格式
    cv2.imwrite(output_path, img)

# 使用函数
remove_border_white_background('./data/meat.jpg', './data/meat.png')


####################
# 方法说明
####################

# 泛洪填充（Flood Fill）是一种图像处理算法，用于将连续的、颜色相近的区域填充为某种颜色。在OpenCV中，cv2.floodFill函数实现了这个功能，其主要参数如下：

# 图像（image）：需要进行泛洪填充的源图像。这个图像应该是可修改的，因为泛洪填充操作会改变图像内容。

# 遮罩（mask）：一个比源图像大一圈的遮罩图像。泛洪填充会在这个遮罩上操作，标记哪些区域已经被填充过。通常，这个遮罩会初始化为全零。

# 种子点（seedPoint）：泛洪填充的开始位置，通常是一个像素点的坐标。填充会从这个点开始，向外扩散。

# 新颜色（newVal）：用于填充区域的颜色。在BGR颜色空间中，这个颜色通常表示为一个三元组，例如(0, 255, 0)表示绿色。

# 低容差（loDiff）和高容差（upDiff）：这两个参数定义了颜色匹配的灵敏度。它们决定了源图像中的哪些颜色将被视为与种子点相近，从而被包括在填充区域内。例如，如果loDiff和upDiff都是(10, 10, 10)，那么源图像中颜色在种子点颜色上下浮动10个单位的像素都会被填充。

# 标志（flags）：控制泛洪填充行为的附加选项，如填充的方式（4连通或8连通）和是否考虑遮罩。

# 在Python中使用OpenCV进行泛洪填充的基本代码示例如下：

    # import cv2
    # import numpy as np

    # img = cv2.imread('image.jpg')
    # h, w = img.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)

    # seed_point = (x, y)  # 泛洪填充的起始点
    # new_color = (255, 0, 0)  # 新颜色，例如红色
    # lo_diff = (10, 10, 10)  # 低容差
    # up_diff = (10, 10, 10)  # 高容差

    # floodfilled = cv2.floodFill(img, mask, seed_point, new_color, lo_diff, up_diff)

# 在这个例子中，floodFill函数将从seed_point指定的位置开始，将所有与种子点颜色相近的区域填充为new_color指定的红色。填充的范围由lo_diff和up_diff控制。
from concurrent.futures import ThreadPoolExecutor
import os
import pyautogui
import time
from PIL import Image, ImageGrab
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import SAM, FastSAM, YOLO


ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
model = FastSAM('FastSAM-s.pt')
model.info()

# load pattern images
pattern1 = cv2.imread('./data/owl1.png', 0)
pattern1_np = np.array(pattern1) # Convert the image to an array
keypoints1, descriptors1 = sift.detectAndCompute(pattern1_np, None)

pattern2 = cv2.imread('./data/owl2.png', 0)
pattern2_np = np.array(pattern2)
keypoints2, descriptors2 = sift.detectAndCompute(pattern2_np, None)

# pattern3 = cv2.imread('./data/box.png', 0)
# pattern3_np = np.array(pattern3)
# keypoints3, descriptors3 = sift.detectAndCompute(pattern3_np, None)

def match_pattern(keypoints_pattern, descriptors_pattern, pattern_np, keypoints_image, descriptors_image, img_np):
    keypoints_image, descriptors_image = sift.detectAndCompute(img_np, None)
    matches = bf.match(descriptors_pattern, descriptors_image)
    matches = sorted(matches, key=lambda x: x.distance)
    N = 50  # 可以根据需要调整这个值
    matched_img = cv2.drawMatches(pattern_np, keypoints_pattern, img_np, keypoints_image, matches[:N], None, flags=2)

    matched_points_pattern = [keypoints_pattern[m.queryIdx].pt for m in matches[:N]]  # 图像1中的匹配关键点位置
    matched_points_image = [keypoints_image[m.trainIdx].pt for m in matches[:N]]  # 图像2中的匹配关键点位置

    if len(matched_points_pattern) < 4 or len(matched_points_image) < 4:
        print("\033[91m匹配点对<4, 不足以计算透视变换\033[0m")
        return None, None, None

    # 如果需要，可以转换成整数坐标
    matched_points_pattern = [tuple(map(int, pt)) for pt in matched_points_pattern]
    matched_points_image = [tuple(map(int, pt)) for pt in matched_points_image]

    # 打印或处理匹配关键点的位置
    # for pt1, pt2 in zip(matched_points1, matched_points2):
    #     print(f"Image 1 point: {pt1} is matched to Image 2 point: {pt2}")

    # 将匹配点转换为float32类型的numpy数组
    pts_pattern = np.float32([keypoints_pattern[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_image = np.float32([keypoints_image[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法计算透视变换矩阵 H
    H, mask = cv2.findHomography(pts_pattern, pts_image, cv2.RANSAC, 5.0)
    
    if H is not None:  # 完整的判断应该是 if not(H is not None and H.shape == (3, 3)):
        identity_matrix = np.eye(3)
        difference = np.abs(H - identity_matrix)
        threshold = 200

        det_H = np.linalg.det(H[0:2, 0:2])
        # 可以根据实际情况调整这里的阈值
        # np.abs(H) 会计算矩阵 H 中每个元素的绝对值。它检查 H 的所有元素的绝对值是否都小于1000。
        # det_H大于0确保映射不是镜像反转, 一个良好的仿射变换的 det(H) 应该是一个正值，而且不应该过大或过小，以避免图像过度压缩或拉伸。
        if np.sum(difference) < threshold and det_H > 0:  
            good_matches = [m for m, good in zip(matches, mask.ravel()) if good]
            pts_image_good = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            centroid = np.mean(pts_image_good, axis=0)
            print(f"\033[94m变换矩阵 H 接近于单位矩阵。Owl形心位置: {centroid} \033[0m")
            pyautogui.click(x=centroid[0][0]+bbox[0]+frame_left, y=centroid[0][1]+bbox[1]+frame_top) 
            time.sleep(0.1)

            temp_img = ImageGrab.grab(bbox=())
            temp_img_np = np.array(temp_img)
            ocr_result_of_owl= ocr.ocr(temp_img_np, cls=True)
            if ocr_result_of_owl[0]:
                if "4级食物"in ocr_result_of_owl[0][0][1][0] or "4级木料"in ocr_result_of_owl[0][0][1][0] or "5级木料"in ocr_result_of_owl[0][0][1][0] or "5级宝箱"in ocr_result_of_owl[0][0][1][0] or "宠物碎片" in ocr_result_of_owl[0][0][1][0]:
                    pyautogui.click(x=857, y=670) 
                    time.sleep(0.1)
                    pyautogui.click(x=685, y=563) 
                    time.sleep(0.1)
                else:
                    pyautogui.click(x=685, y=670) 
                    time.sleep(0.1)
                    pyautogui.click(x=857, y=563) 
                    time.sleep(0.1)
        else:
            print("\033[95m变换矩阵 H 的值异常，可能存在错误匹配。\033[0m")
            return None, None, None
    else:
        print("\033[91m未能找到有效的变换矩阵 H。\033[0m")
        return None, None, None

    # 应用透视变换将原图映射到新图上
    height, width = img_np.shape[:2] # img灰度读取是2维, 彩色读取是3维
    im1Reg = cv2.warpPerspective(pattern_np, H, (width, height)) # 使用 cv2.warpPerspective() 应用透视变换后，im1Reg 变量将包含变换后的图像1（image1），其中只有映射（或变换）的部分有图像的数值，而其他地方的数值通常为0（黑色）
    # 创建一个与图像2大小相同的空白图像
    overlay = np.zeros_like(img_np)

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
    blend = cv2.addWeighted(img_np, beta, overlay, alpha, gamma)

    return matches, matched_img, blend

def is_color_within_range(color1, color2, tolerance):
    """检查 color1 是否在 color2 的指定容差范围内。"""
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))

bbox=(549, 122, 963, 858)
img = ImageGrab.grab(bbox=bbox)  # left, top, right, bottom = bbox, 这里的default位置是游戏的位置
img = ImageGrab.grab()

# Testing code snippet
while True:
    # pyautogui.click(x=655, y=840)
    # time.sleep(0.1)
    img = ImageGrab.grab(bbox=bbox)

    #####################
    # 检查是否有资源采集完成字样, 在某些时候(资源被敌人抢夺完成)会出现屏幕中央出现该字样方框
    #####################
    frame_left = 40
    frame_top = 250
    frame_right = 375
    frame_bottom = 500
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment = img.crop(frame_bbox)
    img_np = np.array(img_fragment)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    ocr_result_center = ocr.ocr(img_np, cls=True)
    print("\033[94m")
    if ocr_result_center[0]:
       print("central notification: ", ocr_result_center[0][0][1][0])
    if ocr_result_center[0] and "采集完成" in ocr_result_center[0][0][1][0]:
        print("central notification: ", ocr_result_center[0][0][1][0])
        pyautogui.click(x=750, y=550) 
        time.sleep(0.1)
        continue
    elif ocr_result_center[0] and "该工人正在你的领地采集" in ocr_result_center[0][0][1][0]:
        print("central notification: ", ocr_result_center[0][0][1][0])
        pyautogui.click(x=680, y=580) 
        time.sleep(0.1)
        pyautogui.click(x=655, y=840)
        time.sleep(0.1)
        continue
    # if ocr_result_center[0]:
    #     pyautogui.click(x=655, y=840)
    #     time.sleep(0.1)

    print("\033[0m")

    #####################
    # 检测猫头鹰
    #####################
    frame_left = 55
    frame_top = 130
    frame_right = 360
    frame_bottom = 595
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment = img.crop(frame_bbox)
    img_np = np.array(img_fragment) # Convert the image to an array

    if img_np.shape[2] == 4:  # RGBA格式
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    keypoints_image, descriptors_image = sift.detectAndCompute(img_np, None)

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(match_pattern, keypoints1, descriptors1, pattern1_np, keypoints_image, descriptors_image, img_np)
        future2 = executor.submit(match_pattern, keypoints2, descriptors2, pattern2_np, keypoints_image, descriptors_image, img_np)
        result1 = future1.result()
        result2 = future2.result()

    #####################
    # 开始对每个column分别进行segmentation和ocr识别, 首先烤ocr识别字块, 然后用segmentation兜底, ocr识字准确率在字小, 背景复杂的情况下效果较差, 需要用segmentation识别分块数量来判断现场有没有资源, 以及资源是否正在被搬运.
    #####################
    for i in range(6):
        # relative frame of img
        frame_left = 55 + i*51
        frame_top = 130
        frame_right = 105 + i*51
        frame_bottom = 595
        frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
        img_fragment = img.crop(frame_bbox)
        img_np = np.array(img_fragment)  # Convert the image to an array
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        ocr_result = ocr.ocr(img_np, cls=True)
        seg_results = model(img_fragment)

        for r in seg_results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            seg_im = np.array(im)
            cv2.imshow(f"Segmentation {i}", seg_im)

        print("\033[96m")
        if ocr_result[0]: 
            # plot text borders
            for line in ocr_result:
                points = line[0][0]  # 边界框坐标
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # 绘制闭合多边形
                cv2.polylines(img_np, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            print("feild resource (first scanned word): ", ocr_result[0][0][1][0])
            cv2.imshow(f"Field Column {i}", img_np)
        else: 
            print("no resource!")
            cv2.imshow(f"Field Column {i}", img_np)
        print("\033[0m")

        # 首要判断条件: 先靠是否识别到字块, 判断资源状态. 
        # 后备方案: 然后通过segmentation的分块数量判断资源状态.
        pixel_color1 = img.getpixel((599-bbox[0], 589-bbox[1]))
        pixel_color2 = img.getpixel((611-bbox[0], 578-bbox[1]))
        print("\033[95m","pixel_color1, pixel_color2: ", pixel_color1, pixel_color2, "\033[0m")
        print("\033[95m","number of segmented objects: ", seg_results[0].boxes.xywhn.shape[0], "\033[0m")
        if ocr_result[0]: 
            flag1 = "LV" in ocr_result[0][0][1][0]
            flag2 = "防" in ocr_result[0][0][1][0]
            flag3 = ":" in ocr_result[0][0][1][0] or (len(ocr_result)>1 and ":" in ocr_result[1][0][1][0])
            if (not flag3) and (flag1 or flag2) and (seg_results[0].boxes.xywhn.shape[0] < 8):
                points = ocr_result[0][0][0] 
                # 计算所有顶点的 x 坐标和 y 坐标的平均值
                average_x = sum(point[0] for point in points) / len(points)
                average_y = sum(point[1] for point in points) / len(points)

                # 创建中心点
                centroid = (average_x, average_y)

                # click centroid
                pyautogui.click(x=centroid[0]+frame_left+bbox[0], y=centroid[1]+frame_top+bbox[1])
                time.sleep(0.2)
                pyautogui.click(x=760, y=580)
                time.sleep(0.2)

                
                temp_img = ImageGrab.grab(bbox=bbox)
                target_color1 = (117, 248, 154, 255)
                target_color2 = (119, 250, 176, 255)
                tolerance = 10  

                pixel_color1 = temp_img.getpixel((599-bbox[0], 589-bbox[1]))
                pixel_color2 = temp_img.getpixel((611-bbox[0], 578-bbox[1]))
                if is_color_within_range(pixel_color1, target_color1, tolerance) and is_color_within_range(pixel_color2, target_color2, tolerance):
                    print("\033[94m","already assigned, exit--------------------", "\033[0m")
                    pyautogui.click(x=860, y=820)
                    time.sleep(0.2)
                    pyautogui.click(x=655, y=840)
                    time.sleep(1.2)
                else:
                    pyautogui.click(x=605, y=585)
                    time.sleep(0.2)
                    pyautogui.click(x=694, y=580)
                    time.sleep(0.2)
                    pyautogui.click(x=860, y=820)
                    time.sleep(0.2)
                    pyautogui.click(x=655, y=840)
                    time.sleep(1.2)
        elif 2 < seg_results[0].boxes.xywhn.shape[0] < 8:
            for box, boxn in zip(r.boxes.xywh, r.boxes.xywhn):
                xn, yn, wn, hn = boxn[:4]
                x, y, w, h = box[:4]
                if wn>0.7 and hn<0.2:
                    pyautogui.click(x=x+frame_left+bbox[0], y=y+frame_top+bbox[1])
                    time.sleep(0.2)
                    pyautogui.click(x=760, y=580)
                    time.sleep(0.2)
                    
                    temp_img = ImageGrab.grab(bbox=bbox)
                    target_color1 = (117, 248, 154, 255)
                    target_color2 = (119, 250, 176, 255)
                    tolerance = 10  

                    pixel_color1 = temp_img.getpixel((599-bbox[0], 589-bbox[1]))
                    pixel_color2 = temp_img.getpixel((611-bbox[0], 578-bbox[1]))
                    if is_color_within_range(pixel_color1, target_color1, tolerance) and is_color_within_range(pixel_color2, target_color2, tolerance):
                        print("\033[94m","already assigned, exit--------------------", "\033[0m")
                        pyautogui.click(x=860, y=820)
                        time.sleep(0.2)
                        pyautogui.click(x=655, y=840)
                        time.sleep(1.2)
                    else:
                        pyautogui.click(x=605, y=585)
                        time.sleep(0.2)
                        pyautogui.click(x=694, y=580)
                        time.sleep(0.2)
                        pyautogui.click(x=860, y=820)
                        time.sleep(0.2)
                        pyautogui.click(x=655, y=840)
                        time.sleep(1.2)
                    break

    # current mouse position
    x, y = pyautogui.position()
    relative_x = x - bbox[0]
    relative_y = y - bbox[1]
    
    # display prompt in stdout
    print("\033[96m")
    if 0 <= relative_x < bbox[2] - bbox[0] and 0 <= relative_y < bbox[3] - bbox[1]:
        # 获取并打印当前鼠标位置的像素颜色
        print("x: ", x, " y: ", y, " relative x:", relative_x, " relative y:", relative_y, " color:", img.getpixel((relative_x, relative_y)))
    else:
        print("x: ", x, " y: ", y, " Mouse is outside the bbox.")
    print("\033[0m")

    # # relative frame of img
    # frame_left = 55
    # frame_top = 130
    # frame_right = 360
    # frame_bottom = 595
    # frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    # img_fragment = img.crop(frame_bbox)
    # img_np = np.array(img_fragment) # Convert the image to an array

    # if img_np.shape[2] == 4:  # RGBA格式
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    # else:
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # keypoints_image, descriptors_image = sift.detectAndCompute(img_np, None)

    # with ThreadPoolExecutor() as executor:
    #     future1 = executor.submit(match_pattern, keypoints1, descriptors1, pattern1_np, keypoints_image, descriptors_image, img_np)
    #     future2 = executor.submit(match_pattern, keypoints2, descriptors2, pattern2_np, keypoints_image, descriptors_image, img_np)
    #     # future3 = executor.submit(match_pattern, keypoints3, descriptors3, pattern3_np, keypoints_image, descriptors_image, img_np)
    #     result1 = future1.result()
    #     result2 = future2.result()
    #     # result3 = future3.result()

    # if result1[0] is None and result2[0] is None:
    #     cv2.imshow('Registered Image 1', img_np)
    #     cv2.imshow('Registered Image 2', img_np)
    #     # cv2.imshow('Registered Image 3', img_np)
    #     continue  # 如果两个线程都失败，则继续循环

    # # 处理匹配成功的情况
    # if result1[0] is not None and result2[0] is not None:
    #     matches1, matched_img_of_pattern1, blend1 = result1
    #     matches2, matched_img_of_pattern2, blend2 = result2
    #     cv2.imshow('Feature Matches 1', matched_img_of_pattern1)
    #     cv2.imshow('Feature Matches 2', matched_img_of_pattern2)
    #     cv2.imshow('Registered Image 1', blend1)
    #     cv2.imshow('Registered Image 2', blend2)
    # elif result1[0] is not None:
    #     matches1, matched_img_of_pattern1, blend1 = result1
    #     cv2.imshow('Feature Matches 1', matched_img_of_pattern1)
    #     cv2.imshow('Registered Image 1', blend1)
    #     cv2.imshow('Registered Image 2', img_np)
    # elif result2[0] is not None:
    #     matches2, matched_img_of_pattern2, blend2 = result2
    #     cv2.imshow('Feature Matches 2', matched_img_of_pattern2)
    #     cv2.imshow('Registered Image 2', blend2)
    #     cv2.imshow('Registered Image 1', img_np)
    # else:
    #     cv2.imshow('Registered Image 2', img_np)
    #     cv2.imshow('Registered Image 1', img_np)
    
    # if result3[0] is not None:
    #     matches3, matched_img_of_pattern3, blend3 = result3
    #     cv2.imshow('Feature Matches 3', matched_img_of_pattern3)
    #     cv2.imshow('Registered Image 3', blend3)
    # else:
    #     cv2.imshow('Registered Image 3', img_np)

    # Wait for 1 millisecond to update the window and check if the user has pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break


####################
# SIFT方法特殊说明
####################
# 这里有个问题, 不管是什么样的image, 当只读取前两维度的pixels信息的时候, 透明度信息消失了, 既然image前两个维度不能设为none, 那么在随后的透视变换依然会变换完整的图像.
# 实际上, 依然有解决方案, 当我们将不用的pixel设为0(黑色)之后, 在透视变换的时候, 就不会去显示那些不要的部分, 而如果设为255(白色)之后, 所有的图像依然会显示.
# 所以这一点, 需要在delete_background.py这个script里设置透明度为0的位置, 三通道值为(0,0,0).
    
# reshape(-1, 1, 2):
# 当应用于一个数组时，reshape(-1, 1, 2) 会改变数组的形状，使得它有三个维度。
# 第二维度的大小被设置为 1，第三维度的大小被设置为 2。
# 第一维度的大小 -1 表示这个维度的大小是由数组的总大小和其他维度的大小来自动确定的。
# 举个例子，假设你有一个形状为 (4, 2) 的数组（即有 4 行 2 列），当你调用 reshape(-1, 1, 2) 时，结果将是一个形状为 (4, 1, 2) 的数组。这意味着插了一个维度。

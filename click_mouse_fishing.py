import os
import pyautogui
import time
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
import random


####################
# Functionalities:
# 1. 自动钓鱼
# 2. 自动开坐骑
####################

# while True:
#     x, y = pyautogui.position()
#     print(f"X: {x}, Y: {y}")

# 645, 722
# fishing return: 600 +- 10, 828 +- 10
# field 915 +- 10, 790 +- 10
# field return 587 +- 10, 810 +- 10
# fishing 915 +- 10, 672 +- 10

####################
# Fishing
####################
# while True:
#     img = ImageGrab.grab()
#     pyautogui.click(x=675, y=672)  # 2. 自动开坐骑
#     x, y = pyautogui.position()  # 获取当前鼠标位置
#     pixel_color = img.getpixel((x, y))
#     # pyautogui.moveTo(583, 355) 
#     print(f"X: {x}, Y: {y}, Colour: {pixel_color}")
#     time.sleep(1.2)  # 每秒输出一次位置

ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
process_tag = False
breads_tag = True
breads_nonce = 10

# Testing code snippet
while True:
    left, top, right, bottom = (549, 122, 963, 858)
    bbox=(left, top, right, bottom)
    img = ImageGrab.grab(bbox=bbox) # left, top, right, bottom = bbox, 这里的default位置是游戏的位置
    # pyautogui.click(x=710, y=762)

    # relative frame of img 1
    frame_left = 100
    frame_top = 390
    frame_right = 320
    frame_bottom = 430
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment1 = img.crop(frame_bbox)
    
    # relative frame of img 2
    frame_left = 280
    frame_top = 450
    frame_right = 390
    frame_bottom = 480
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment2 = img.crop(frame_bbox)

    # relative frame of img 3
    frame_left =140  # 100
    frame_top = 485  # 620
    frame_right = 310  # 180
    frame_bottom = 535  # 645
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment3 = img.crop(frame_bbox)

    # relative frame of img 4
    frame_left = 601 - left
    frame_top = 193 - top
    frame_right = 662 - left
    frame_bottom = 215 - top
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment4 = img.crop(frame_bbox)

    # relative frame of img 5
    frame_left = 644 - left
    frame_top = 740 - top
    frame_right = 722 - left
    frame_bottom = 762 - top
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment5 = img.crop(frame_bbox)

    # relative frame of img 6
    frame_left = 734 - left
    frame_top = 740 - top
    frame_right = 779 - left
    frame_bottom = 760 - top
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment6 = img.crop(frame_bbox)

    # relative frame of img 7
    frame_left = 810 - left
    frame_top = 548 - top
    frame_right = 858 - left
    frame_bottom = 571 - top
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment7 = img.crop(frame_bbox)

    img_np1 = np.array(img_fragment1) # Convert the image to an array
    img_np2 = np.array(img_fragment2) # Convert the image to an array
    img_np3 = np.array(img_fragment3) # Convert the image to an array
    img_np4 = np.array(img_fragment4) # Convert the image to an array
    img_np5 = np.array(img_fragment5) # Convert the image to an array
    img_np6 = np.array(img_fragment6) # Convert the image to an array
    img_np7 = np.array(img_fragment7) # Convert the image to an array
    result1 = ocr.ocr(img_np1, cls=True)
    result2 = ocr.ocr(img_np2, cls=True)
    result3 = ocr.ocr(img_np3, cls=True)
    result4 = ocr.ocr(img_np4, cls=True)
    result5 = ocr.ocr(img_np5, cls=True)
    result6 = ocr.ocr(img_np6, cls=True)
    result7 = ocr.ocr(img_np7, cls=True)
    frame1 = cv2.cvtColor(img_np1, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame2 = cv2.cvtColor(img_np2, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame3 = cv2.cvtColor(img_np3, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame4 = cv2.cvtColor(img_np4, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame5 = cv2.cvtColor(img_np5, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame6 = cv2.cvtColor(img_np6, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame7 = cv2.cvtColor(img_np7, cv2.COLOR_BGR2RGB) # Convert the color to RGB

    # 设置中文文字的位置和字体, 只能使用PIL
    font_path = "./SimHei.ttf" # 替换为您的中文字体文件路径
    font_size = 12
    font_color1 = (255, 255, 255) 
    font_color2 = (255, 255, 255) 
    font_color3 = (255, 255, 255) 
    image1_pil = Image.fromarray(frame1)
    image2_pil = Image.fromarray(frame2)
    image3_pil = Image.fromarray(frame3)
    draw1 = ImageDraw.Draw(image1_pil)
    draw2 = ImageDraw.Draw(image2_pil)
    draw3 = ImageDraw.Draw(image3_pil)
    font = ImageFont.truetype(font_path, font_size)
    text1 =  f"detected:\n{result1[0][0][1][0]}" if result1[0] else "" 
    text2 =  f"detected:\n{result2[0][0][1][0]}" if result2[0] else "" 
    text3 =  ("Pro: True", (0, 255, 0)) if process_tag else ("Pro: False", (0, 0, 255))
    position1 = (0, 5)  # 文字的位置坐标 (x, y)
    position2 = (0, 3)  # 文字的位置坐标 (x, y)
    position3 = (50, 3)  # 文字的位置坐标 (x, y)
    draw1.text(position1, text1, font=font, fill=font_color1)
    draw2.text(position2, text2, font=font, fill=font_color2)
    draw3.text(position3, text3[0], font=font, fill=text3[1])
    frame1 = np.array(image1_pil)
    frame2 = np.array(image2_pil)
    frame3 = np.array(image3_pil)

    # # 如果只输出英文, 可以直接使用cv2
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # position = (0, 15)
    # font_scale = 1
    # font_color = (255, 255, 255) 
    # thickness = 2
    # cv2.putText(frame1, text, position, font, font_scale, font_color, thickness)

    window_name1 = "Fish/Bycatch Name"
    window_name2 = "Object Feature"
    window_name3 = "Sale Check"
    window_name4 = "Breads Number"
    window_name5 = "Sell Fish or not"
    window_name6 = "Sell Bycatch or not?"
    window_name7 = "Sell Comfirmation"
    cv2.imshow(window_name1, frame1) # Show the first image in an OpenCV window
    cv2.moveWindow(window_name1, 0, 100) 
    cv2.imshow(window_name2, frame2) # Show the second image in another OpenCV window
    cv2.moveWindow(window_name2, 0, 200)
    cv2.imshow(window_name3, frame3) 
    cv2.moveWindow(window_name3, 0, 300)
    cv2.imshow(window_name4, frame4) 
    cv2.moveWindow(window_name4, 0, 400)
    cv2.imshow(window_name5, frame5) 
    cv2.moveWindow(window_name5, 0, 500)
    cv2.imshow(window_name6, frame6) 
    cv2.moveWindow(window_name6, 0, 600)
    cv2.imshow(window_name7, frame7) 
    cv2.moveWindow(window_name7, 0, 700)
    # Wait for 1 millisecond to update the window and check if the user has pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

    if result5[0]: 
        print("result 5", result5[0][0][1][0])
    else: 
        print("No result 5")
    if result6[0]: 
        print("result 6", result6[0][0][1][0])
    else: 
        print("No result 6")
    if result7[0]: 
        print("result 7", result7[0][0][1][0])
    else: 
        print("No result 7")
    # continue

    # Decide the next steps based on the current number of bread.
    if result4[0]: 
        breads_info = result4[0][0][1][0]
        try:
            # Trying to extract and convert numbers
            left_breads = int(breads_info.split('/')[0])
        except ValueError:
            # If the conversion fails, print an error message and set a default value for left_breads to avoid program stopping
            print("Invalid input, unable to convert to int.")
            left_breads = 1
        print("current breads number:  ", left_breads, " tag: ", breads_tag)
        if result1[0] or result2[0] or result3[0]:
            pass
        elif left_breads == 0:
            breads_tag = False
            print("current breads nonce: ", breads_nonce)
            continue
        elif breads_tag is False and left_breads < 10 + breads_nonce:
            print("current breads nonce: ", breads_nonce)
            continue
        else: 
            breads_tag = True
            breads_nonce = random.randint(0, 10)
    else:
        print("Breads detection fails")

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
    
    if result1[0] and result2[0] and result3[0]: 
        print("detected word: ", result1[0][0][1][0], ", ", result2[0][0][1][0], ", ", result3[0][0][1][0])
    elif result1[0] and result2[0]:
        print("detected word: ", result1[0][0][1][0], ", ", result2[0][0][1][0])
    elif result1[0] and result3[0]:
        print("detected word: ", result1[0][0][1][0], ", ", result3[0][0][1][0])
    elif result1[0]: 
        print("detected word: ", result1[0][0][1][0])
    elif result3[0]: 
        print("detected word: ", result3[0][0][1][0])
    elif result5[0] and (result5[0][0][1][0] == "出售" or "交" in result5[0][0][1][0]): pass
    elif result6[0] and result6[0][0][1][0] == "出售": pass
    elif result7[0] and result7[0][0][1][0] == "确定": pass
    else: 
        print("No detected words \033[0m") 
        avoid_x_start, avoid_x_end = 770, 890
        avoid_y_start, avoid_y_end = 730, 785
        target_x = 720
        target_y = 700
        while True:
            offset_x = random.randint(-80, 100)  # x偏移量范围
            offset_y = random.randint(-100, 100)  # y偏移量范围
            new_x = target_x + offset_x
            new_y = target_y + offset_y
            if not (avoid_x_start <= new_x <= avoid_x_end and avoid_y_start <= new_y <= avoid_y_end):
                break 
        nonce = random.random()
        if process_tag == False:
            process_tag = True
            pyautogui.click(x=target_x + offset_x, y=target_y + offset_y)
        elif process_tag == True and nonce < 0.2:
            process_tag = True
            pyautogui.click(x=target_x + offset_x, y=target_y + offset_y)
        time.sleep(0.1 + random.uniform(0, 0.5))
        continue
    print("\033[0m")

    if result1[0] and (result1[0][0][1][0] == "吞噬鳗" or result1[0][0][1][0] == "东星斑" or result1[0][0][1][0] == "虎鲨" or result1[0][0][1][0] == "血鹦鹉" or result1[0][0][1][0] == "天竺鲷" or result1[0][0][1][0] == "剑尾鱼" or result1[0][0][1][0] == "帝王蟹" or result1[0][0][1][0] == "兰寿金鱼"):
        if result2[0] and "新纪录" in result2[0][0][1][0]:
            break
    if result1[0] and (result1[0][0][1][0] == "灵魂鱼" or result1[0][0][1][0] == "玩具鲨" or result1[0][0][1][0] == "拟态章鱼" or result1[0][0][1][0] == "钻石鱼"):
        break
    if result2[0] and ("首次" in result2[0][0][1][0]):
        break

    if result5[0] and (result5[0][0][1][0] == "出售" or "交" in result5[0][0][1][0]) :
        offset_x = random.randint(-30, 30)  # x偏移量范围
        offset_y = random.randint(-15, 15)  # y偏移量范围
        pyautogui.click(x=683 + offset_x, y=750 + offset_y)
    if result6[0] and result6[0][0][1][0] == "出售":
        offset_x = random.randint(-30, 30)  # x偏移量范围
        offset_y = random.randint(-15, 15)  # y偏移量范围
        pyautogui.click(x=755 + offset_x, y=750 + offset_y)
    if result7[0] and result7[0][0][1][0] == "确定":
        offset_x = random.randint(-30, 30)  # x偏移量范围
        offset_y = random.randint(-15, 15)  # y偏移量范围
        pyautogui.click(x=834 + offset_x, y=560 + offset_y)

    time.sleep(1.2 + random.uniform(0, 0.5))
    process_tag = False

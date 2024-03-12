import os
import pyautogui
import time
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
import random


# functionality;
# 1. 自动钓鱼
# 2. 自动开坐骑

bbox=(549, 122, 963, 858)
img = ImageGrab.grab(bbox=bbox)  # left, top, right, bottom = bbox, 这里的default位置是游戏的位置
img = ImageGrab.grab()

####################
# Fishing
####################
# while True:
#     # pyautogui.click(x=710, y=762)  # 1. 自动钓鱼
#     # pyautogui.click(x=710, y=762)  # 1. 自动钓鱼
#     pyautogui.click(x=675, y=672)  # 2. 自动开坐骑
#     x, y = pyautogui.position()  # 获取当前鼠标位置
#     pixel_color = img.getpixel((x, y))
#     # pyautogui.click(x=820, y=580)  # 1. 自动钓鱼
#     print(img.getpixel((1200, 355)))
#     # pyautogui.moveTo(583, 355) 
#     # x, y = pyautogui.position()
#     # if img.getpixel((x,y)) == (110, 91, 61, 255): 
#         # print(img.getpixel((x,y)))
#     print(f"X: {x}, Y: {y}, Colour: {pixel_color}")
#     time.sleep(1.2)  # 每秒输出一次位置


ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
process_tag = False

# Testing code snippet
while True:
    img = ImageGrab.grab(bbox=bbox)
    # pyautogui.click(x=710, y=762)

    # relative frame of img
    frame_left = 100
    frame_top = 390
    frame_right = 320
    frame_bottom = 430
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment1 = img.crop(frame_bbox)
    
    # relative frame of img
    frame_left = 280
    frame_top = 450
    frame_right = 390
    frame_bottom = 480
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment2 = img.crop(frame_bbox)

    # relative frame of img
    frame_left =140  # 100
    frame_top = 490  # 620
    frame_right = 310  # 180
    frame_bottom = 540  # 645
    frame_bbox = (frame_left, frame_top, frame_right, frame_bottom)
    img_fragment3 = img.crop(frame_bbox)

    img_np1 = np.array(img_fragment1) # Convert the image to an array
    img_np2 = np.array(img_fragment2) # Convert the image to an array
    img_np3 = np.array(img_fragment3) # Convert the image to an array
    result1 = ocr.ocr(img_np1, cls=True)
    result2 = ocr.ocr(img_np2, cls=True)
    result3 = ocr.ocr(img_np3, cls=True)
    frame1 = cv2.cvtColor(img_np1, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame2 = cv2.cvtColor(img_np2, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame3 = cv2.cvtColor(img_np3, cv2.COLOR_BGR2RGB) # Convert the color to RGB

    # 设置中文文字的位置和字体, 只能使用PIL
    font_path = "./SimHei.ttf" # 替换为您的中文字体文件路径
    font_size = 12
    font_color1 = (255, 255, 255) 
    font_color2 = (255, 255, 255) 
    image1_pil = Image.fromarray(frame1)
    image2_pil = Image.fromarray(frame2)
    draw1 = ImageDraw.Draw(image1_pil)
    draw2 = ImageDraw.Draw(image2_pil)
    font = ImageFont.truetype(font_path, font_size)
    text1 =  f"detected:\n{result1[0][0][1][0]}" if result1[0] else "" 
    text2 =  f"detected:\n{result2[0][0][1][0]}" if result2[0] else "" 
    position1 = (0, 5)  # 文字的位置坐标 (x, y)
    position2 = (0, 3)  # 文字的位置坐标 (x, y)
    draw1.text(position1, text1, font=font, fill=font_color1)
    draw2.text(position2, text2, font=font, fill=font_color2)
    frame1 = np.array(image1_pil)
    frame2 = np.array(image2_pil)

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
    cv2.imshow(window_name1, frame1) # Show the first image in an OpenCV window
    cv2.moveWindow(window_name1, 0, 100) 
    cv2.imshow(window_name2, frame2) # Show the second image in another OpenCV window
    cv2.moveWindow(window_name2, 0, 200)
    cv2.imshow(window_name3, frame3) # Show the second image in another OpenCV window
    cv2.moveWindow(window_name3, 0, 300)
    # Wait for 1 millisecond to update the window and check if the user has pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
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
    else: 
        print("No detected words \033[0m") 
        offset_x = random.randint(-100, 100)  # x偏移量范围
        offset_y = random.randint(-100, 100)  # y偏移量范围
        nonce = random.random()
        if process_tag == False:
            process_tag = True
            pyautogui.click(x=740 + offset_x, y=680 + offset_y)
        elif process_tag == True and nonce < 0.2:
            process_tag = True
            pyautogui.click(x=740 + offset_x, y=680 + offset_y)
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

    offset_x = random.randint(-10, 10)  # x偏移量范围
    offset_y = random.randint(-10, 10)  # y偏移量范围
    pyautogui.click(x=720 + offset_x, y=762 + offset_y)
    time.sleep(0.5 + random.uniform(0, 0.5))
    offset_x = random.randint(-10, 10)  # x偏移量范围
    offset_y = random.randint(-10, 10)  # y偏移量范围
    pyautogui.click(x=820+ offset_x, y=580 + offset_y)
    time.sleep(1.2 + random.uniform(0, 0.5))
    process_tag == False

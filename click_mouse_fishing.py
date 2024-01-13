import os
import pyautogui
import time
from PIL import ImageGrab
import cv2
import numpy as np
from paddleocr import PaddleOCR


# functionality;
# 1. 自动钓鱼
# 2. 自动开坐骑

bbox=(549, 122, 963, 858)
img = ImageGrab.grab(bbox=bbox)  # left, top, right, bottom = bbox, 这里的default位置是游戏的位置
img = ImageGrab.grab()

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

# Testing code snippet
while True:
    img = ImageGrab.grab(bbox=bbox)
    # pyautogui.click(x=710, y=762)

    # relative frame of img
    frame_left = 100
    frame_top = 400
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

    img_np1 = np.array(img_fragment1) # Convert the image to an array
    img_np2 = np.array(img_fragment2) # Convert the image to an array
    result1 = ocr.ocr(img_np1, cls=True)
    result2 = ocr.ocr(img_np2, cls=True)
    frame1 = cv2.cvtColor(img_np1, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    frame2 = cv2.cvtColor(img_np2, cv2.COLOR_BGR2RGB) # Convert the color to RGB
    cv2.imshow("Screen Capture 1", frame1) # Show the first image in an OpenCV window
    cv2.imshow("Screen Capture 2", frame2) # Show the second image in another OpenCV window
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
    if result1[0] and result2[0]: 
        print("detected word: ", result1[0][0][1][0], ", ", result2[0][0][1][0])
    elif result1[0]: 
        print("detected word: ", result1[0][0][1][0])
    else: 
        print("No detected words \033[0m") 
        pyautogui.click(x=710, y=650)
        continue
    print("\033[0m")

    if result1[0] and (result1[0][0][1][0] == "鲟鱼" or result1[0][0][1][0] == "虎鲨" or result1[0][0][1][0] == "血鹦鹉" or result1[0][0][1][0] == "天竺鲷" or result1[0][0][1][0] == "帝王蟹" or result1[0][0][1][0] == "兰寿金鱼"):
        if result2[0] and "新纪录" in result2[0][0][1][0]:
            break
    if result1[0] and (result1[0][0][1][0] == "灵魂鱼" or result1[0][0][1][0] == "玩具鲨"):
        break
    if result2[0] and ("首次" in result2[0][0][1][0]):
        break

    pyautogui.click(x=710, y=762)
    time.sleep(0.1)
    pyautogui.click(x=820, y=580)
    time.sleep(1.2)


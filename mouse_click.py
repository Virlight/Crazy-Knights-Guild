import pyautogui
import time
from PIL import ImageGrab

# functionality;
# 1. 自动钓鱼
# 2. 自动开坐骑

img = ImageGrab.grab()

while True:
    pyautogui.click(x=710, y=762)  # 1. 自动钓鱼
    pyautogui.click(x=710, y=762)  # 1. 自动钓鱼
    # pyautogui.click(x=675, y=672)  # 2. 自动开坐骑
    x, y = pyautogui.position()  # 获取当前鼠标位置
    pixel_color = img.getpixel((x, y))
    pyautogui.click(x=820, y=580)  # 1. 自动钓鱼
    # print(img.getpixel((1200, 355)))
    # pyautogui.moveTo(583, 355) 
    # x, y = pyautogui.position()
    # if img.getpixel((x,y)) == (110, 91, 61, 255): 
        # print(img.getpixel((x,y)))
    print(f"X: {x}, Y: {y}, Colour: {pixel_color}")
    time.sleep(1)  # 每秒输出一次位置√
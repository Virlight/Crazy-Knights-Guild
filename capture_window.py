import os
import pywinctl as pwc
import subprocess
from PIL import Image

# subprocess.Popen(['open', '-a', 'Notes']) # 启动该文件, 从标准程序目录

# allwindows = pwc.getAllTitles() # 检查目前已经打开的所有windows标题
# print(allwindows);quit()

windows = pwc.getWindowsWithTitle('疯狂骑士团', condition=pwc.Re.CONTAINS, flags=pwc.Re.IGNORECASE)
if windows:
    win = windows[0]
    # win.resizeTo(600, 400, wait=True) # 在左上角位置不变的情况下, 通过两个参数(length, height)调整window size
    frame = win.getClientFrame()  # 仅仅得到那个框的位置, 如Rect(left=222, top=182, right=1443, bottom=810), 不含任何图像信息
    print(frame)

    screenshot_path = os.path.join(os.getcwd(), "test_screenshot.png")
    # 一般情况下，screencapture 会将截图保存为文件，但在这个命令中：-o 告诉 screencapture 省略窗口阴影。- 表示输出将被发送到标准输出流（stdout），而不是写入磁盘上的文件。
    # capture_command = f"screencapture -R{frame.left},{frame.top},{frame.right-frame.left},{frame.bottom-frame.top} -o -"
    # process = subprocess.Popen(capture_command, shell=True, stdout=subprocess.PIPE)
    # image_data, _ = process.communicate()   
    # img = Image.open(io.BytesIO(image_data)); img.show()

    capture_command = f"screencapture -R{frame.left},{frame.top},{frame.right-frame.left},{frame.bottom-frame.top} {screenshot_path}"
    subprocess.call(capture_command, shell=True)
    img = Image.open(screenshot_path)
    img.show()

else:
    print("Window not found. Check application name and language")
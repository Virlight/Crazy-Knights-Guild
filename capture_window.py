from datetime import datetime
import io
import os
import time
import pywinctl as pwc
import subprocess
from PIL import ImageGrab, Image
import pyautogui
import cv2
import numpy as np

# subprocess.Popen(['open', '-a', 'Notes']) # 启动该文件, 从标准程序目录

# allwindows = pwc.getAllTitles() # 检查目前已经打开的所有windows标题
# print(allwindows)
TEST_WINDOW_TITLE = "疯狂骑士团"

def is_daytime():
    """判断当前是否为白天"""
    hour = datetime.now().hour
    return 7 <= hour <= 20

def get_icon_path(icon_name):
    """根据当前时间和图标是否有反色版本，返回相应的图标路径"""
    day_time = is_daytime()
    
    # 如果图标有正反色版本，根据时间选择
    if isinstance(icon_name, dict):
        return icon_name["day"] if day_time else icon_name["night"]
    # 否则，直接返回图标路径
    return icon_name

def activate_mac_window(window_title):
    windows = pwc.getWindowsWithTitle(window_title, condition=pwc.Re.CONTAINS, flags=pwc.Re.IGNORECASE)
    if windows:
        win = windows[0]
        win.activate()
    else:
        print("Window not found. Check application name and language")

def click_icon_within_window(window_list, icon_list):
    for i in range(len(window_list)):
        window_title, icon_index = window_list[i]
        windows = pwc.getWindowsWithTitle(window_title, condition=pwc.Re.CONTAINS, flags=pwc.Re.IGNORECASE)
        if windows:
            win = windows[0]
            win.activate()
            # win.resizeTo(600, 400, wait=True) # 在左上角位置不变的情况下, 通过两个参数(length, height)调整window size
            frame = win.getClientFrame()  # 仅仅得到那个框的位置, 如Rect(left=222, top=182, right=1443, bottom=810), 不含任何图像信息
            print(frame)

            next_icon_index = window_list[i+1][1] if i+1 < len(window_list) else len(icon_list)
            for icon_path in icon_list[icon_index:next_icon_index]:
                print(f"Trying to such {icon_path} ...")
                while not click_icon(icon_path, frame):
                    print(f"Icon not found. Retry to such {icon_path} ...")
                    time.sleep(1)
                time.sleep(1)

def test_window(window_title):
    windows = pwc.getWindowsWithTitle(window_title, condition=pwc.Re.CONTAINS, flags=pwc.Re.IGNORECASE)
    if windows:
        win = windows[0]
        win.activate()
        # win.resizeTo(600, 400, wait=True) # 在左上角位置不变的情况下, 通过两个参数(length, height)调整window size
        frame = win.getClientFrame()  # 仅仅得到那个框的位置, 如Rect(left=222, top=182, right=1443, bottom=810), 不含任何图像信息
        print(frame)

        # 不采用储存的方式打开Window Frame图片 (失败, 无法打开图片): 
        # 一般情况下，screencapture 会将截图保存为文件，但在这个命令中：-o 告诉 screencapture 省略窗口阴影。- 表示输出将被发送到标准输出流（stdout），而不是写入磁盘上的文件。
        # capture_command = f"screencapture -R{frame.left},{frame.top},{frame.right-frame.left},{frame.bottom-frame.top} -o -"
        # process = subprocess.Popen(capture_command, shell=True, stdout=subprocess.PIPE)
        # image_data, _ = process.communicate()   
        # img = Image.open(io.BytesIO(image_data)); img.show()
        
        # 采用储存的方式打开Window Frame图片, 使用电脑自带的截屏命令 (有效, 可储存图片, 保留高清原图, 特征点更多, 但是图片尺寸会有变化):
        screenshot_path = os.path.join(os.getcwd(), "data", "test_screenshot.png")
        capture_command = f"screencapture -R{frame.left},{frame.top},{frame.right-frame.left},{frame.bottom-frame.top} {screenshot_path}"
        subprocess.call(capture_command, shell=True)
        img = Image.open(screenshot_path) # 此时的img尺寸有变换, 只能使用img.size[0]和img.size[1]来获取宽和高
        img_np = np.array(img)

        # 使用Pillow从当前桌面截取Window Frame位置的图片(有效, 不用调尺寸, 但是图片像素更低, 与我通过在window上截取的图片的像素不一样, 导致特征点不匹配):
        # bbox=(frame.left, frame.top, frame.right, frame.bottom)
        # img = ImageGrab.grab(bbox=bbox)
        # img_np = np.array(img)

        icon_np = cv2.imread("data/mini_projects_icon.jpg")
        click_icon(icon_np, img_np, frame, img.size)
        icon_np = cv2.imread("data/game_icon.jpg")
        click_icon(icon_np, img_np, frame, img.size)

    else:
        print("Window not found. Check application name and language")

def click_icon(icon_name, frame):
    screenshot_path = os.path.join(os.getcwd(), "data", "tmp_image.png")
    capture_command = f"screencapture -R{frame.left},{frame.top},{frame.right-frame.left},{frame.bottom-frame.top} {screenshot_path}"
    subprocess.call(capture_command, shell=True)
    img = Image.open(screenshot_path) # 此时的img尺寸有变换, 只能使用img.size[0]和img.size[1]来获取宽和高
    img_np = np.array(img)

    icon_np = cv2.imread(icon_name)
    return _click_icon(icon_np, img_np, frame, img.size)

def _click_icon(icon, target, frame, img_size):
    # Covnert icon and target images to grayscale
    icon_img = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    target_img = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the icon and target images
    icon_kp, icon_desc = sift.detectAndCompute(icon_img, None)
    target_kp, target_desc = sift.detectAndCompute(target_img, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors using KNN match
    matches = bf.knnMatch(icon_desc, target_desc, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(f"Number of good matches: {len(good_matches)}")

    # Find inliers using homography
    if len(good_matches) > 4:
        icon_pts = np.float32([icon_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        target_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(icon_pts, target_pts, cv2.RANSAC, 5.0)

        # Remove duplicate points
        if H is None:
            unique_icon_pts, icon_id = np.unique(icon_pts, axis=0, return_index=True)
            unique_target_pts, _ = np.unique(target_pts, axis=0, return_index=True)
            unique_good_matches = [good_matches[i] for i in icon_id]
            if len(unique_good_matches) == 3: 
                M = cv2.getAffineTransform(unique_icon_pts, unique_target_pts)
                theta = np.arctan2(M[0, 1], M[0, 0])
                print(f"Rotation angle of affine transformation: {theta}")

                # 使用掩码（mask）选择内点（inliers: 正确匹配的点）
                matchesMask = [0] * len(good_matches)
                for id in icon_id:
                    matchesMask[id] = 1

                # 绘制内点
                draw_params = dict( # matchColor = (0,255,0), # 画匹配点的颜色
                                singlePointColor = None,
                                matchesMask = matchesMask, # 画出内点
                                flags = 2)

                matched_img = cv2.drawMatches(icon_img, icon_kp, target_img, target_kp, good_matches, None, **draw_params)
                
                # cv2.moveWindow('Feature Matches', 0, 0)
                # cv2.imshow('Feature Matches', matched_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                check_angles = [0, np.pi/2 , np.pi]
                allowed_tolerance = np.pi / 180 # 1 degrees
                match_found = any(abs(abs(theta) - angle) <= allowed_tolerance for angle in check_angles)

                if match_found: 
                    unique_good_matches_pts = np.float32([target_kp[m.trainIdx].pt for m in unique_good_matches])
                    centroid = np.mean(unique_good_matches_pts, axis=0)

                    click_x, click_y = centroid[0], centroid[1]
                    click_x_frame = click_x * (frame.right - frame.left) / img_size[0] + frame.left
                    click_y_frame = click_y * (frame.bottom - frame.top) / img_size[1] + frame.top
                    print(f"Clicking on ({click_x_frame}, {click_y_frame})")
                    pyautogui.click(x=click_x_frame, y=click_y_frame)
                    return True
                else:
                    print("Icon not found in the target image, because of large rotation")
                    return False
            else:
                print("Icon not found in the target image, because of less points")
                return False

        # Get inlier matches    
        inlier_matches = [m for m, msk in zip(good_matches, mask) if msk]

        # 使用掩码（mask）选择内点（inliers: 正确匹配的点）
        matchesMask = mask.ravel().tolist()

        # 绘制内点
        draw_params = dict( # matchColor = (0,255,0), # 画匹配点的颜色
                        singlePointColor = None,
                        matchesMask = matchesMask, # 画出内点
                        flags = 2)

        matched_img = cv2.drawMatches(icon_img, icon_kp, target_img, target_kp, good_matches, None, **draw_params)
        
        # cv2.moveWindow('Feature Matches', 0, 0)
        # cv2.imshow('Feature Matches', matched_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        theta = np.arctan2(H[0, 1], H[0, 0])
        print(f"Rotation angle of perspective transformation: {theta}")

        # Calculate the rotation angle of inlier matches
        check_angles = [0, np.pi/2 , np.pi]
        allowed_tolerance = np.pi / 18 # 10 degrees
        match_found = any(abs(abs(theta) - angle) <= allowed_tolerance for angle in check_angles)

        if len(inlier_matches) > 0 and match_found: 
            inlier_pts = np.float32([target_kp[m.trainIdx].pt for m in inlier_matches])
            centroid = np.mean(inlier_pts, axis=0)

            click_x, click_y = centroid[0], centroid[1]
            click_x_frame = click_x * (frame.right - frame.left) / img_size[0] + frame.left
            click_y_frame = click_y * (frame.bottom - frame.top) / img_size[1] + frame.top
            print(f"Clicking on ({click_x_frame}, {click_y_frame})")
            pyautogui.click(x=click_x_frame, y=click_y_frame)
            return True
        else:
            print("Icon not found in the target image, because of large rotation")
            return False
    else:
        print("Icon not found in the target image")
        return False

if __name__ == "__main__":
    allwindows = pwc.getAllTitles()
    print(allwindows)
    window_list = [('wechat (chats)', 0), ('疯狂骑士团', 2)] # window title, first icon index for this window
    icon_list = [
        {"day": "data/mini_projects_icon_day.jpg", "night": "data/mini_projects_icon_night.jpg"}, 
        "data/game_icon.jpg", 
        "data/start_game_button.jpg", 
        "data/close_button.jpg", 
        "data/fish_field_button.jpg"]
    click_icon_within_window(window_list, icon_list)

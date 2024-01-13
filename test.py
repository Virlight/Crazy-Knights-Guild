import os
import subprocess
import time

print("Current Working Directory:", os.getcwd())

time.sleep(1)

screenshot_path = os.path.join(os.getcwd(), "test_screenshot.png")
subprocess.call(f"screencapture {screenshot_path}", shell=True)
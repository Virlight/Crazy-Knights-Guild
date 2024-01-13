import cv2

img_path = './data/owl1.jpg'
output_path = './data/owl1.png'

img = cv2.imread(img_path)

if img.shape[2] == 3:  # 如果是RGB格式，转换为RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

cv2.imwrite(output_path, img)

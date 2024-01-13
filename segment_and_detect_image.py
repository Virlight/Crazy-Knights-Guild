import os
from ultralytics import SAM, FastSAM, YOLO
from PIL import Image

# Load a model
# model = SAM('sam_b.pt')
# model = SAM('mobile_sam.pt')
model = FastSAM('FastSAM-s.pt')
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8n-seg.pt')
model.info() # Display model information (optional)
results = model('./data/column5.png') # Run inference

for r in results:
    print(r.boxes.xywhn)
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('./data/results.jpg')  # save image



##################
# Effect comparision
##################
# 效果比较：
# 准确性：从准确性角度来说, SAM原始方法是最优的, 其次是mobilesam和FastSAM, 而Yolo的segmentation是最弱的
# 速度：速度最快的应该是yolo的segment, 速度在100ms之内, 其次是FastSAM, 这两个速度都在500之内, 随后是mobilesam和sam, mobile可能更快一点, 但都是和sam同一个两集over 50000 ms
# 用法: 1. 其实不管怎样, 直接的目标检测一定比segmentation快, 比如yolo一定比yolo-seg快, 因为segmemt需要对所有像素点进行分类.
#      2. 对于SAM, 其实它会根据你截图的范围, 标出你图像中更像是一个object的区域, 所以如果你可能尽可能缩小你想要的obj的背景范围, result会更好. 
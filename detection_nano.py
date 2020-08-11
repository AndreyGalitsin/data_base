import jetson.inference
import jetson.utils
import cv2

def object_detection(net, img):
    img, width, height = jetson.utils.loadImageRGBA(img)
    detections = net.Detect(img, width, height)
    img_numpy = jetson.utils.cudaToNumpy(img, width, height, 4)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./img.jpg', img_numpy)
    return img_numpy

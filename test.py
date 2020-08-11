import pyrebase
import base64
from PIL import Image
from base64 import decodestring
import cv2
import pickle
import numpy as np
import io
import datetime

import torchvision.models.detection
#from detecto.core import Model
from visualize_detection import detect_live
import torch

from detection_nano import object_detection

import jetson.inference


if __name__ == "__main__":
    model = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.8)
    img = cv2.imread('./test.jpeg')
    detected_img = object_detection(model, img)
    cv2.imwrite('./photo.jpg', detected_img)
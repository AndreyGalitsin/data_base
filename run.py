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

class DBase():
    def __init__(self):

        firebaseConfig = {
                            "apiKey": "AIzaSyDK9B9NCjSwJJL2ryiy7USK1XRIOzKoh5M",
                            "authDomain": "nirs-camera.firebaseapp.com",
                            "databaseURL": "https://nirs-camera.firebaseio.com",
                            "projectId": "nirs-camera",
                            "storageBucket": "nirs-camera.appspot.com",
                            "messagingSenderId": "382258061836",
                            "appId": "1:382258061836:web:bab2a86d0a88a57cc5c429",
                            "measurementId": "G-VNNLQ9Y1NN"
                            }
        firebase = pyrebase.initialize_app(firebaseConfig)
        self.db = firebase.database()

        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        print('device', device)
        #self.model = Model(device=device)
        self.model = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.8)

    def object_detection_detecto(self, image_ndarray):
        detected_image = detect_live(self.model, image = image_ndarray, score_filter=0.6)
        return detected_image

    def object_detection_nano(self, image_ndarray):
        path = './photo.jpg'
        cv2.imwrite(path, image_ndarray)
        detected_image = object_detection(self.model, './test.jpeg')
        return detected_image

    def send_data(self, detected_image):
        detected_image_base64 = self.convert_to_base64(detected_image)
        self.db.child("image").update({"1after": detected_image_base64})
        self.db.child("image").update({"1before": self.data})

    def read_data(self):
        ip_cam = self.db.child("image/ipcamnum").get().val()
        print(ip_cam)

        self.data = self.db.child("image/" + ip_cam + "/data").get().val()
        data = self.data.split(',')
        self.first_name = data[0]
        image = data[1]
        image = self.convert_to_image(image)
        return image

    def show_cv2(self, image_ndarray):
        cv2.imshow("Image", image_ndarray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_image(self, base64_code):
        imgdata = base64.b64decode(str(base64_code))
        image = Image.open(io.BytesIO(imgdata))
        image_ndarray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        #self.show_cv2(image_ndarray)
        return image_ndarray

    def convert_to_base64(self, image):
        retval, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode("utf-8") 
        image_base64 = self.first_name + ',' + image_base64
        return image_base64

    def run(self):
        while 1:
            t1=datetime.datetime.now()
            image = self.read_data()
            detected_image = self.object_detection_nano(image) 
            #self.show_cv2(detected_image)
            self.send_data(detected_image)
            t2=datetime.datetime.now()
            print('DONE!', t2-t1)


if __name__ == "__main__":
    dbase=DBase()
    dbase.run()
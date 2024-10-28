import common as cm # type: ignore #! 이 게 멀 까?
import cv2 as cv
import numpy as np
import util as ut # type: ignore
ut.init_gpio()

import time
import sys
sys.path.insert(0, '/var/www/html/earthover')

from PIL import Image
from threading import Thread

# 웹캠으로 실시간 촬영
cap = cv.VideoCapture(0)

# 임계값 설정
threshold = 0.2

# number of objects to be shown as detected
top_k = 1

# model
model_dir = '/var/www/html/all_models'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

# tolerance
tolerance = 0.1
x_deviation = 0
y_deviation = 0
arr_track_data = [0, 0, 0, 0, 0, 0]

arr_valid_objects = ['qrcode']


#!----- FastAPI? network ------

#!----- Dobot movement ------

#!----- object tracking ------
def track_object(objs, labels):

    # global delay
    global x_deviation, y_deviation, tolerance, arr_track_data

    if (len(objs)==0):
        print("no objects to track..!")
        ut.stop()
        ut.red_light("OFF")
        arr_track_data = [0, 0, 0, 0, 0, 0]
        return
    
    # ut.head_lights("OFF")
    k = 0
    flag = 0
    for obj in objs:
        lbl = labels.get(obj.id, obj.id)
        k = arr_valid_objects.count(lbl)
        
        if (k > 0):
            x_min, y_min, x_max, y_max = list(obj.bbox)
            flag = 1
            break
    
    if (flag == 0):
        print("selected object no present")
        return
    
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    # round 함수는 앞의 숫자를 뒤에 숫자만큼 반올림하는 역할
    print("x_diff :", round(x_diff, 5))
    print("y_diff :", round(y_diff, 5))

    obj_x_center = x_min + (x_diff / 2)
    obj_x_center = round(obj_x_center, 3)

    obj_y_center = y_min + (y_diff / 2)
    obj_y_center = round(obj_y_center, 3)


    x_deviation = round(0.5 - obj_x_center, 3)
    y_deviation = round(0.5 - obj_y_center, 3)

    print("{", x_deviation, y_deviation, "}")
    
    # move dobot
    thread = Thread(target=move_robot)

    # thread 시작
    thread.start()
    # thread.join()

    #print(cmd)

    arr_track_data[0] = obj_x_center
    arr_track_data[1] = obj_y_center
    arr_track_data[2] = x_deviation
    arr_track_data[3] = y_deviation


#def move_robot():
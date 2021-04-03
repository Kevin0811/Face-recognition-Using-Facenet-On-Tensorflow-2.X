from itertools import count
import cv2 
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import time
import threading

import paho.mqtt.client as mqtt
import json  
import datetime 


confidence_t=0.99
recognition_t=0.8
required_size = (160,160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img ,detector,encoder,encoding_dict):
    name = None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'--{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img ,name

def mqtt_init():
    # 連線設定
    # 初始化地端程式
    client = mqtt.Client()

    # 設定登入帳號密碼
    client.username_pw_set("kevin","0811")

    # 設定連線資訊(IP, Port, 連線時間)
    client.connect("192.168.31.247", 1883, 60)

    return client


def motion_init(frame):
    avg_img = cv2.blur(frame, (4, 4))
    avg_float = np.float32(avg_img)
    return avg_float, avg_img

def motion(frame, avg_float, avg_img, count=1, time=3, sensitive=0.1):

    different_area = 0
    frame_h ,frame_w, _ = frame.shape
    # 模糊處理
    blur = cv2.blur(frame, (4, 4))

    # 計算目前影格與平均影像的差異值
    diff = cv2.absdiff(avg_img, blur)

    # 將圖片轉為灰階
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 篩選出變動程度大於門檻值的區域
    ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # 使用型態轉換函數去除雜訊
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 產生等高線
    thresh = np.uint8(thresh)
    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 更新平均影像
    cv2.accumulateWeighted(blur, avg_float, 0.02)
    avg_img = cv2.convertScaleAbs(avg_float)

    if cnts is not None:
        # 畫出等高線（除錯用）
        cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
        
        for c in cnts:
            different_area+=cv2.contourArea(c)
        if different_area/(frame_h*frame_w) > sensitive:
            return True, avg_float, avg_img, count, time

    return False, avg_float, avg_img, count, time

# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
		
	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame

    def getstatue(self):
	# 傳最新狀態。
        return self.status

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()


if __name__ == "__main__":
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    is_moving = False
    user_name = None
    
    #cap = cv2.VideoCapture(0)
    URL = 'rtsp://192.168.31.18:554/unicast'
    # 連接攝影機
    ipcam = ipcamCapture(URL)

    # 啟動子執行緒
    ipcam.start()

    # 暫停1秒，確保影像已經填充
    time.sleep(1)
    #ret, frame = video.read()

    if ipcam.getstatue():
        #print("ipcamera work\n")
        frame = ipcam.getframe()
        avg_float, avg_img = motion_init(frame)
        client = mqtt_init()
        client.publish("kevinpc/motion", "OFF")
        cv2.namedWindow("Motion Camera", cv2.WINDOW_AUTOSIZE)
        start_time = time.time()
        count_moving = 0
    else:
        print("CAM NOT OPEND\n")

    while True:

        current_time = time.time()

        pass_time = current_time - start_time

        if pass_time > 10:
            start_time = current_time
            count_moving = 0
            client.publish("kevinpc/motion", "OFF")
        elif count_moving > 1:
            client.publish("kevinpc/motion", "ON")
            count_moving = 0
            start_time = current_time
            continue
        
        frame = ipcam.getframe()

        is_moving, avg_float, avg_img, _, _ = motion(frame, avg_float, avg_img)

        if is_moving:
            frame , user_name = detect(frame , face_detector , face_encoder , encoding_dict)
            
            if user_name=="unknown":
                #client.publish("kevinpc/motion", "ON")
                start_time = current_time
                count_moving+=1
            else:
                if user_name:
                    print("Detect User : " + str(user_name))
                #client.publish("kevinpc/motion", "OFF")

        frame_scale = cv2.resize(frame, (960, 540))
        cv2.imshow('Motion Camera', frame_scale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            ipcam.stop()
            break

    cv2.destroyAllWindows()
    



import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import time


cap = cv2.VideoCapture(0)
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('127.0.0.1', 8083))

print(current_user.username)

start=time.time()

while (time.time()-start)<=140:
    ret,frame = cap.read()
    cv2.imshow('Recording frame',frame)
    cv2.waitKey(1)
    data = pickle.dumps(frame)
    clientsocket.sendall(struct.pack("L", len(data)) + data)

cap.release()
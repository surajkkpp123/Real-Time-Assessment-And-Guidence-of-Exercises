import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import time


cap = cv2.VideoCapture('input.mp4')
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('127.0.0.1',12347))


start=time.time()

while (time.time()-start)<=120:
    ret,frame = cap.read()
    #cv2.imshow('Recording frame',frame)
    #cv2.waitKey(1)
    data = pickle.dumps(frame)
    clientsocket.sendall(struct.pack("L", len(data)) + data)

cap.release()
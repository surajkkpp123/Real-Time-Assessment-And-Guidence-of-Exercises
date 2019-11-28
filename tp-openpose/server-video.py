import socket
import sys
import cv2
import pickle
import numpy as np
import struct
import time

HOST = '127.0.0.1'
PORT = 8083

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()
out = cv2.VideoWriter('input.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (640, 480))
data = b''
payload_size = struct.calcsize("L")

start=time.time()

while (time.time()-start)<=110:
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]

    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
   
    frame=pickle.loads(frame_data)
    out.write(frame)
    print('---------------------Dimensions--------------------------')
    print(frame.shape[1])
    print(frame.shape[0])
    print('----------------------Dimensions-------------------------')
    print(time.time()-start)
    print(frame.size)

out.release()
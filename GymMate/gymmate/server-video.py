import socket
import sys
import cv2
import pickle
import numpy as np
import struct 
import time
import zlib
import os
import shutil


HOST=''

PORT=12347



s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

print('Socket created')



s.bind((HOST,PORT))

print('Socket bind complete')

s.listen(10)

print('Socket now listening')



conn,addr=s.accept()

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (640, 480))


data = b""

payload_size = struct.calcsize(">L")

print("payload_size: {}".format(payload_size))


print('Output Stream Created')

start=time.time()

while (time.time()-start)<=60:

    while len(data) < payload_size:

        print("Recv: {}".format(len(data)))

        data += conn.recv(4096)


    if len(data)==0:
    	break



    print("Done Recv: {}".format(len(data)))

    packed_msg_size = data[:payload_size]

    data = data[payload_size:]

    msg_size = struct.unpack(">L", packed_msg_size)[0]

    print("msg_size: {}".format(msg_size))

    while len(data) < msg_size:

        data += conn.recv(4096)
        if len(data)==0:
            break

    frame_data = data[:msg_size]

    data = data[msg_size:]



    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")

    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    out.write(frame)
    print('---------------------Dimensions--------------------------')
    print(frame.shape[1])
    print(frame.shape[0])
    print('----------------------Dimensions-------------------------')
    print(time.time()-start)
    #cv2.imshow('ImageWindow',frame)

    cv2.waitKey(1)


out.release()



try:
    path1=r"C:\Users\ragha\Desktop\Full deployement Code\Gym Mate\output.mp4"
    path2=r"C:\Users\ragha\Desktop\Full deployement Code\Gym Mate\gymmate\static\output.mp4"
    os.rename(path1,path2)
    shutil.move(path1,path2)
    os.replace(path1,path2)

except:
	print("An exception occured")
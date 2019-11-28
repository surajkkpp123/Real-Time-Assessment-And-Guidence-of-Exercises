import argparse
import logging
import time
import os
import cv2
import numpy as np


import tf_pose.estimator 
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from gtts import gTTS 
 

def put_text(image,str,b,g,r):
    cv2.putText(image,
                str,
                (5, 100),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (b, g, r), 2)


def play_sound(mytext):
    myobj = gTTS(text=mytext, lang='en', slow=False) 
    myobj.save("welcome.mp3")
    os.system("mpg321 welcome.mp3")

if __name__=='__main__':

    logger = logging.getLogger('TfPoseEstimator-WebServer')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fps_time = 0
    model='mobilenet_thin'
    resize='432x368'
    resize_out_ratio=4.0
    camera=1
    type='r_bicep_curl'


    print(tf_pose.estimator.typearg)
    tf_pose.estimator.typearg = type
    print(tf_pose.estimator.typearg)
    typearg = tf_pose.estimator.typearg
   
   

    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh(resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    logger.debug('cam read+')

    cam = cv2.VideoCapture(1)
#######################################################
    # play_sound('welcome to the pose trainer')

    

    ret_val, image = cam.read()#image is a frame here, ret_val is either true or false
    # image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (image.shape[1], image.shape[0]))

    print('------------------Video Dimensions-------------------------------')

    print(image.shape[1])
    print(image.shape[0])
    
    print('------------------Video Dimensions--------------------------------')
    r_bicep_curl = []
    prev_ang=-1
    direction="none"

    timerr = 0
    maxx_time = 0
    
    while True:
        ret_val, image = cam.read()
        
        if typearg=='push_ups' or typearg=='high_plank':
            image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if ret_val == False:
            break

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        logger.debug('postprocess+')
        image,ans = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        print("anssssssssssssssssssssssssssssssssss")
        print(ans)

        if typearg=='push_ups' or typearg=='high_plank':
            image=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        if ans!="no" and len(ans)==2 and (typearg=='r_bicep_curl' or typearg=='l_bicep_curl'):
            ans[0] = round(ans[0])
            ans[1] = round(ans[1])
            if ans[1] <=10:
                if prev_ang!=-1:
                    cv2.putText(image,
                        'prev_angle = %d' % (prev_ang),
                        (100, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                    
                    if ans[0]==prev_ang:
                        put_text(image,'GOING CORRECT: ',0,255,0)

                    

                    elif ans[0]<prev_ang and direction=='none' and abs(ans[0]-prev_ang)>=3:
                        direction='up'

                    elif ans[0]<prev_ang and direction=='down' and abs(ans[0]-prev_ang)>=3:
                        direction='up'
                        if prev_ang<=160:
                            print('you are not fully bringing the forearm down')
                            #play_sound('you are not fully bringing the forearm down')
                            put_text(image,'INCORRECT: you are not fully bringing the forearm down',0,0,255)

                    elif ans[0]>prev_ang and direction=='none' and abs(ans[0]-prev_ang)>=3:
                        direction='down'

                    elif ans[0]>prev_ang and direction=='up' and abs(ans[0]-prev_ang)>=3 :
                        direction='down'
                        if prev_ang>=40:
                            print('you are not fully bringing the forearm up')
                            #play_sound('you are not fully bringing the forearm up')
                            put_text(image,'INCORRECT: you are not fully bringing the forearm up',0,0,255)
                    else:
                        put_text(image,'GOING CORRECT: ',0,255,0)
                    if abs(ans[0]-prev_ang)>=3:
                        prev_ang=ans[0]
                    cv2.putText(image,
                        direction,
                        (25, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                

                else: 
                    prev_ang=ans[0]
        
                    
            else:
                print('your torso and upper arm are moving a lot relative to each other')
               # play_sound('your torso and upper arm are moving a lot relative to each other')
                put_text(image,'INCORRECT: your torso and upper arm are moving a lot relative to each other',0,0,255)
                if prev_ang!=-1 and abs(ans[0]-prev_ang)>=3:
                   prev_ang=ans[0] 
                else:
                    prev_ang=ans[0]
 

            cv2.putText(image,
                        "angles : %d %d" % (ans[0],ans[1]),
                        (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)

        elif ans!="no" and len(ans)==4 and typearg == 'push_ups':############################################
            ans[0]=round(ans[0])
            ans[1]=round(ans[1])
            ans[2]=round(ans[2])
            ans[3]=round(ans[3])
           
            if prev_ang!=-1 and ans[2]>=150 and ans[3]>=150:
                if ans[0]==prev_ang:
                    # print("you are not moving your body")
                        #play_sound("you are not moving the arm at all")
                    put_text(image,'GOING CORRECT: ',0,255,0)

                elif ans[0]<prev_ang and direction=='none' and abs(ans[0]-prev_ang)>3:
                    direction='down'

                elif ans[0]<prev_ang and direction=='up' and abs(ans[0]-prev_ang)>3:
                    direction='down'
                    if prev_ang<170:
                        print('you are not raising your body fully up and hands are not fully straight')
                        put_text(image,'INCORRECT: you are not raising your body fully up and hands are not fully straight',0,0,255)

                elif ans[0]>prev_ang  and direction=='none' and abs(ans[0]-prev_ang)>3:
                    direction='up'

                elif  ans[0]>prev_ang and direction=='down' and abs(ans[0]-prev_ang)>3:
                    direction='up'
                    if prev_ang>60:
                        print('you are not lowering your body fully down and hands are not fully bend')
                        put_text(image,'INCORRECT: you are not lowering your body fully down',0,0,255)
                else:
                        put_text(image,'GOING CORRECT: ',0,255,0)

                if abs(ans[0]-prev_ang)>3:
                    prev_ang=ans[0]

                cv2.putText(image,
                        direction,
                        (25, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

            elif prev_ang==-1 and ans[2]>150 and ans[3]>150 :
                prev_ang = ans[0]    
            
            else:
                if ans[2]<150:
                    print('INCORRECT: your torso and thigh are not in a straight line')
                    put_text(image,'INCORRECT: your torso and thigh are not in a straight line',0,0,255)

                elif ans[3]<150:
                    print('your legs are not straight')
                    put_text(image,'INCORRECT: your legs are not straight',0,0,255)

                
        elif ans!="no" and len(ans)==4 and typearg == 'high_plank':
            ans[0]=round(ans[0])
            ans[1]=round(ans[1])
            ans[2]=round(ans[2])
            ans[3]=round(ans[3])

            if ans[0]>=160 and ans[1]>=30 and ans[2]>=160 and ans[3]>=160:
                put_text(image,'GOING CORRECT',0,255,0)
                timerr+=1
                maxx_time = max(maxx_time,timerr)

            elif timerr==0:
                put_text(image,'you are not ready',0,255,0)
                
            elif timerr>0:
                timerr=0
                cv2.putText(image,
                        "You held a plank for %d seconds at most" % (maxx_time),
                        (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
            








        else:
            cv2.putText(image,
                    "angles: not detected",
                    (5, 200),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 154), 2)    

        cv2.imshow('GYM trainer result', image)
        out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    cam.release()    
    out.release()
    cv2.destroyAllWindows()

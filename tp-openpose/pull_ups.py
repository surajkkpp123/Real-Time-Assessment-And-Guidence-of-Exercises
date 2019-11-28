import logging
import time

import cv2
import numpy as np

import tf_pose.estimator
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# In[2]:


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# # Declaring Variables

# In[7]:


fps_time = 0 
model='mobilenet_thin'
resolution='432x368'
resize_out_ratio=4.0
typearg='pull_ups'
showBG=True
video='input.mp4'



print(tf_pose.estimator.typearg)
tf_pose.estimator.typearg = typearg
print(tf_pose.estimator.typearg)
typearg = tf_pose.estimator.typearg


# In[8]:


logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
w, h = model_wh(resolution)
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
cap = cv2.VideoCapture(video)


# In[9]:


if cap.isOpened() is False:
    print("Error opening video stream or file")
else:
    print('File Opened Correctly')


# In[19]:


out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (640,480))
r_bicep_curl = []
prev_ang=-1
direction="none"
timerr = 0
maxx_time = 0


# # Functions Required

# In[20]:


def put_text(image,str,b,g,r):
    cv2.putText(image,str,(5,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(b,g,r),2)


# In[21]:


def play_sound():
    playsound('welcome.mp3')


# # Evaluation

# In[23]:


while True:
    ret_val, image = cap.read()
        
    if typearg=='push_ups' or typearg=='high_plank':
        image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if ret_val == False:
        break

    logger.debug('image process+')
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    logger.debug('postprocess+')
    image,ans = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    print("---------------------------ans-----------------------------")
    print(ans)

    if typearg=='push_ups' or typearg=='high_plank':
        image=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
    logger.debug('show+')
    cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)



    if ans!="no" and len(ans)==4 and typearg == 'push_ups':
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

    elif ans!="no" and len(ans)==2 and (typearg=='pull_ups'):
        ans[0] = round(ans[0])
        ans[1] = round(ans[1])
        if ans[1]<=180:
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
                        print('you are not fully bringing your foerarms down')
                        #play_sound('you are not fully bringing the forearm down')
                        put_text(image,'INCORRECT: you are not fully bringing your forearms down',0,0,255)

                elif ans[0]>prev_ang and direction=='none' and abs(ans[0]-prev_ang)>=3:
                    direction='down'

                elif ans[0]>prev_ang and direction=='up' and abs(ans[0]-prev_ang)>=3 :
                    direction='down'
                    if prev_ang>=40:                    
                        print('you are not fully bringing your forearms up')
                            #play_sound('you are not fully bringing the forearm up')
                        put_text(image,'INCORRECT: you are not fully bringing your forearms up',0,0,255)
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
    
cap.release()    
out.release()
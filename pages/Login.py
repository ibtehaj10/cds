import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import pandas as pd 
import torch 
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image
import pandas as pd
import torch
import base64
import streamlit.components.v1 as components
import csv
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import os
import csv
from streamlit_option_menu import option_menu
#  x-x-x-x-x-x-x-x-x-x-x-x-x-x LOGIN FORM x-x-x-x-x-x-x-x-x

from datetime import timedelta        
import streamlit as st
import pandas as pd
import hashlib
import sqlite3 
#

import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import pyautogui

# print("Done !!!")

data = ["student Count",'Date','Id']
with open('final.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(data)
    

# # l1 = []
# # l2 = []
# # if st.button('signup'):
    
    
# #     usernames = st.text_input('Username')
# #     pwd = st.text_input('Password') 
# #     l1.append(usernames)
# #     l2.append(pwd)

# #     names = ["dmin", "ser"]
# #     if st.button("signupsss"):
# #         username =l1

# #         password =l2

# #         hashed_passwords =stauth.Hasher(password).generate()

# #         file_path = Path(__file__).parent / "hashed_pw.pkl"

# #         with file_path.open("wb") as file:
# #             pickle.dump(hashed_passwords, file)
            
    
# # elif st.button('Logins'):
# names = ['dmin', 'ser']

# username = []

# file_path = Path(__file__).parent / 'hashed_pw.pkl'

# with file_path.open('rb') as file:
#     hashed_passwords = pickle.load(file)

# authenticator = stauth.Authenticate(names,username,hashed_passwords,'Cheating Detection','abcdefg',cookie_expiry_days=180)

# name,authentication_status,username= authenticator.login('Login','main')


# if authentication_status == False:
#     st.error('Username/Password is incorrect')
    
# if authentication_status == None:
#     st.error('Please enter a username and password')

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


#img = get_img_as_base64("/home/anas/PersonTracking/WebUI/attendence.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.xmple.com/wallpaper/blue-gradient-black-linear-1920x1080-c2-87cefa-000000-a-180-f-14.svg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
files = pd.read_csv('LoginStatus.csv')


idS = list(files['Id'])
Pwd = list(files['Password'].astype(str))

# print(type(Pwd))
ids = st.sidebar.text_input('Enter a username')
Pswd = st.sidebar.text_input('Enter a password',type="password",key="password")
btn = st.sidebar.button('Login')
# print('list : ',type(Pwd))

    
if ids and Pswd !=  "":      
    if (ids in idS) and(str(Pswd) in Pwd):
        
            # st.empty()    
            date_time = time.strftime("%b %d %Y %-I:%M %p")
            date = date_time.split()
            dates = date[0:3]
            times = date[3:5]
            # x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-xAPPLICACTION -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x

            def non_max_suppression_fast(boxes, overlapThresh):
                try:
                    if len(boxes) == 0:
                        return []

                    if boxes.dtype.kind == "i":
                        boxes = boxes.astype("float")

                    pick = []

                    x1 = boxes[:, 0]
                    y1 = boxes[:, 1]
                    x2 = boxes[:, 2]
                    y2 = boxes[:, 3]

                    area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    idxs = np.argsort(y2)

                    while len(idxs) > 0:
                        last = len(idxs) - 1
                        i = idxs[last]
                        pick.append(i)

                        xx1 = np.maximum(x1[i], x1[idxs[:last]])
                        yy1 = np.maximum(y1[i], y1[idxs[:last]])
                        xx2 = np.minimum(x2[i], x2[idxs[:last]])
                        yy2 = np.minimum(y2[i], y2[idxs[:last]])

                        w = np.maximum(0, xx2 - xx1 + 1)
                        h = np.maximum(0, yy2 - yy1 + 1)

                        overlap = (w * h) / area[idxs[:last]]

                        idxs = np.delete(idxs, np.concatenate(([last],
                                                            np.where(overlap > overlapThresh)[0])))

                    return boxes[pick].astype("int")
                except Exception as e:
                    print("Exception occurred in non_max_suppression : {}".format(e))


            protopath = "MobileNetSSD_deploy.prototxt"
            modelpath = "MobileNetSSD_deploy.caffemodel"
            detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
            # Only enable it if you are using OpenVino environment
            # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


            CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]

            tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

            st.markdown(
                """
                <style>
                [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
                    width: 350px
                }
                [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
                    width: 350px
                    margin-left: -350px
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)


            # Resize Images to fit Container
            @st.cache()
            # Get Image Dimensions
            def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
                dim = None
                (h,w) = image.shape[:2]

                if width is None and height is None:
                    return image

                if width is None:
                    r = width/float(w)
                    dim = (int(w*r),height)

                else:
                    r = width/float(w)
                    dim = width, int(h*r)

                # Resize image
                resized = cv.resize(image,dim,interpolation=inter)

                return resized
                    
            # About Page
            # authenticator.logout('Logout')
            EXAMPLE_NO = 3


            def streamlit_menu(example=1):
                if example == 1:
                    # 1. as sidebar menu
                    with st.sidebar:
                        selected = option_menu(
                            menu_title="Main Menu",  # required
                            options=["Home", "Projects", "Contact"],  # required
                            icons=["house", "book", "envelope"],  # optional
                            menu_icon="cast",  # optional
                            default_index=0,  # optional
                        )
                    return selected

                if example == 2:
                    # 2. horizontal menu w/o custom style
                    selected = option_menu(
                        menu_title=None,  # required
                        options=["Home", "Projects", "Contact"],  # required
                        icons=["house", "book", "envelope"],  # optional
                        menu_icon="cast",  # optional
                        default_index=0,  # optional
                        orientation="horizontal",
                    )
                    return selected

                if example == 3:
                    # 2. horizontal menu with custom style
                    selected = option_menu(
                        menu_title=None,  # required
                        options=[ "Projects"],  # required
                        icons=["house", "book", "envelope"],  # optional
                        menu_icon="cast",  # optional
                        default_index=0,  # optional
                        orientation="horizontal",
                        styles={
                            "container": {"padding": "0!important", "background-color": "#eaeaea"},
                            "icon": {"color": "#080602", "font-size": "18px"},
                            "nav-link": {
                                "font-size": "18px",
                                "text-align": "left",
                                "color": "#000000",
                                "margin": "0px",
                                "--hover-color": "#E1A031",
                            },
                            "nav-link-selected": {"background-color": "#ffffff"},
                        },
                    )
                    return selected


            selected = streamlit_menu(example=EXAMPLE_NO)

            # if selected == "Home":
            #     st.title(f"You have selected {selected}")
            # if selected == "Projects":
            #     st.title(f"You have selected {selected}")
            # if selected == "Contact":
            #     st.title(f"You have selected {selected}")
            # app_mode = st.sidebar.selectbox(
            #                 'App Mode',
            #                 ['Application']
            #                 )
            if selected == 'Projects':
            # 2. horizontal menu with custom style
                # selected = option_menu(
                #     menu_title=None,  # required
                #     options=["Home", "Projects", "Contact"],  # required
                #     icons=["house", "book", "envelope"],  # optional
                #     menu_icon="cast",  # optional
                #     default_index=0,  # optional
                #     orientation="horizontal",
                #     styles={
                #         "container": {"padding": "0!important", "background-color": "#fafafa"},
                #         "icon": {"color": "orange", "font-size": "25px"},
                #         "nav-link": {
                #             "font-size": "25px",
                #             "text-align": "left",
                #             "margin": "0px",
                #             "--hover-color": "#eee",
                #         },
                #         "nav-link-selected": {"background-color": "blue"},
                #     },
                # )
            # if app_mode == 'About':
            #     st.title('About Product And Team')
            #     st.markdown('''
            #                 Imran Bhai Project
            #     ''')
            #     st.markdown(
            #         """
            #         <style>
            #         [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            #             width: 350px
            #         }
            #         [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            #             width: 350px
            #             margin-left: -350px
            #         }
            #         </style>
            #         """,
            #         unsafe_allow_html=True,
            #     )

                


            # elif app_mode == 'Application':
                
                st.set_option('deprecation.showfileUploaderEncoding', False)

                use_webcam = "pass"
                # record = st.sidebar.checkbox("Record Video")

                # if record:
                #     st.checkbox('Recording', True)

                # drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

                # st.sidebar.markdown('---')

                # ## Add Sidebar and Window style
                # st.markdown(
                #     """
                #     <style>
                #     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
                #         width: 350px
                #     }
                #     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
                #         width: 350px
                #         margin-left: -350px
                #     }
                #     </style>
                #     """,
                #     unsafe_allow_html=True,
                # )

                # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value=1)
                # st.sidebar.markdown('---')
                # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
                # tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)
                # st.sidebar.markdown('---')

                ## Get Video
                stframe = st.empty()
                video_file_buffer = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
                temp_file = tempfile.NamedTemporaryFile(delete=False)

                
                if not video_file_buffer:
                    if use_webcam:
                        video = cv.VideoCapture(0)
                    else:
                        try:
                            video = cv.VideoCapture(1)
                            temp_file.name = video
                        except:
                            pass
                else:
                    temp_file.write(video_file_buffer.read())
                    video = cv.VideoCapture(temp_file.name)

                width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps_input = int(video.get(cv.CAP_PROP_FPS))

                ## Recording
                codec = cv.VideoWriter_fourcc('a','v','c','1')
                out = cv.VideoWriter('output1.mp4', codec, fps_input, (width,height))

                # st.sidebar.text('Input Video')
                # st.sidebar.video(temp_file.name)

                fps = 0
                i = 0

                drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

                kpil2, kpil3,kpil4,kpil5 = st.columns(4)

                

                with kpil2:
                    st.markdown('**STUDENT ID**')
                    kpil2_text = st.markdown('0')

                with kpil3:
                    st.markdown('**Mobile**')
                    kpil3_text = st.markdown('0')
                with kpil4:
                    st.markdown('**Watch**')
                    kpil4_text = st.markdown('0')
                with kpil5:
                    st.markdown('**Count**')
                    kpil5_text = st.markdown('0')

                


                st.markdown('<hr/>', unsafe_allow_html=True)
                # try:
                def main():
                    db = {}
                    
                    # cap = cv2.VideoCapture('//home//anas//PersonTracking//WebUI//movement.mp4')
                    path='yolo0vs5/yolov5s-int8.tflite'
                    #count=0
                    custom = 'yolov5s'

                    model = torch.hub.load('yolovs5', custom, path,source='local',force_reload=True)

                    b=model.names[0] = 'person'
                    mobile = model.names[67] = 'cell phone'
                    watch = model.names[75] = 'clock'

                    fps_start_time = datetime.datetime.now()
                    fps = 0
                    size=416

                    count=0
                    counter=0


                    color=(0,0,255)

                    cy1=250
                    offset=6


                    pt1 = (120, 100)
                    pt2 = (980, 1150)
                    color = (0, 255, 0)

                    pt3 = (283, 103)
                    pt4 = (1500, 1150)
                    
                    cy2 = 500
                    color = (0, 255, 0)
                    total_frames = 0
                    frame_no = 0
                    prevTime = 0
                    cur_frame = 0
                    count=0
                    counter=0
                    fps_start_time = datetime.datetime.now()
                    fps = 0
                    total_frames = 0
                    lpc_count = 0
                    opc_count = 0
                    object_id_list = []
                    # success = True
                    if st.button("Detect"):
                        try:
                            while video.isOpened():
                                current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)

                                # Convert milliseconds to seconds
                                current_time_s = current_time_ms / 1000

                                # Create a timedelta object from the total number of seconds
                                current_time = timedelta(seconds=current_time_s)

                                # Extract hours, minutes, and seconds from the timedelta object
                                hours, remainder = divmod(current_time.seconds, 3600)
                                minutes, seconds = divmod(remainder, 60)

                                # Print the current time stamp
                                timez = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
                                ret, frame = video.read()
                                frame = imutils.resize(frame, width=600)
                                total_frames = total_frames + 1
                            
                                (H, W) = frame.shape[:2]
                                # if frame:
                                #     print("for frame : " + str(frame_no) + "   timestamp is: ", str(video.get(cv2.CAP_PROP_POS_MSEC)))
                                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

                                detector.setInput(blob)
                                person_detections = detector.forward()
                                rects = []
                                for i in np.arange(0, person_detections.shape[2]):
                                    confidence = person_detections[0, 0, i, 2]
                                    if confidence > 0.5:
                                        idx = int(person_detections[0, 0, i, 1])

                                        if CLASSES[idx] != "person":
                                            continue

                                        person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                                        (startX, startY, endX, endY) = person_box.astype("int")
                                        rects.append(person_box)

                                boundingboxes = np.array(rects)
                                boundingboxes = boundingboxes.astype(int)
                                rects = non_max_suppression_fast(boundingboxes, 0.3)

                                objects = tracker.update(rects)
                                for (objectId, bbox) in objects.items():
                                    x1, y1, x2, y2 = bbox
                                    x1 = int(x1)
                                    y1 = int(y1)
                                    x2 = int(x2)
                                    y2 = int(y2)
                                    #print()
                                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    text = "ID: {}".format(objectId+1)
                                    # print(text)1
                                    cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                                    if objectId not in object_id_list:
                                        object_id_list.append(objectId)
                                fps_end_time = datetime.datetime.now()
                                time_diff = fps_end_time - fps_start_time
                                if time_diff.seconds == 0:
                                    fps = 0.0
                                else:
                                    fps = (total_frames / time_diff.seconds)
                                    # print(fps)

                                fps_text = "FPS: {:.2f}".format(fps)

                                cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                                lpc_count = len(objects)
                                opc_count = len(object_id_list)

                                lpc_txt = "PC: {}".format(lpc_count)
                                # print(lpc_txt)
                                opc_txt = "OPC: {}".format(opc_count)
                                
                                count += 1
                                if count % 4 != 0:
                                    continue
                                # frame=cv.resize(frame, (600,500))
                                # cv2.line(frame, pt1, pt2,color,2)
                                # cv2.line(frame, pt3, pt4,color,2)
                                results = model(frame,size)
                                components = results.pandas().xyxy[0]
                                for index, row in results.pandas().xyxy[0].iterrows():
                                    x1 = int(row['xmin'])
                                    y1 = int(row['ymin'])
                                    x2 = int(row['xmax'])
                                    y2 = int(row['ymax'])
                                    confidence  = (row['confidence'])
                                    obj = (row['class'])

                                    
                                    # min':x1,'ymin':y1,'xmax':x2,'ymax':y2,'confidence':confidence,'Object':obj}
                                    # if lpc_txt is not None:
                                    # 	try:
                                    # 		db["student Count"] = [lpc_txt]
                                    # 	except:
                                    # 		db["student Count"] = ['N/A']
                                    if obj == 0:
                                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                                        rectx1,recty1 = ((x1+x2)/2,(y1+y2)/2)
                                        rectcenter = int(rectx1),int(recty1)
                                        cx = rectcenter[0]
                                        cy = rectcenter[1]
                                        cv2.circle(frame,(cx,cy),3,(0,255,0),-1)
                                        cv2.putText(frame,str(b), (x1,y1), cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                        
                                        # db["student Count"] = [lpc_txt]
                                        # db['Date'] = [date_time]
                                        # db['id'] = ['N/A']
                                        # db['Mobile']=['N/A']
                                        # db['Watch'] = ['N/A']
                                        if cy<(cy1+offset) and cy>(cy1-offset):
                                            DB = []
                                            counter+=1
                                            DB.append(counter)

                                            ff = DB[-1]
                                            fx = str(ff)
                                            # cv2.line(frame, pt1, pt2,(0, 0, 255),2)
                                            # if cy<(cy2+offset) and cy>(cy2-offset):

                                            # cv2.line(frame, pt3, pt4,(0, 0, 255),2)
                                            font = cv2.FONT_HERSHEY_TRIPLEX
                                            cv2.putText(frame,fx,(50, 50),font, 1,(0, 0, 255),2,cv2.LINE_4)
                                            cv2.putText(frame,"Movement",(70, 70),font, 1,(0, 0, 255),2,cv2.LINE_4)
                                            kpil2_text.write(f"<h5 style='text-align: left; color:white;'>{text}</h5>", unsafe_allow_html=True)
                                            db["student Count"] = [lpc_txt]
                                            db['Date'] = [timez]
                                            
                                            db['id'] = [text]
                                            name = "screenshot/"+str(timez) + '.jpg'
                                            print ('Creating...' + name)
                                            db['Image Path']=name
                                            cv2.imwrite(name, frame)
                                            
                                            # myScreenshot = pyautogui.screenshot()
                                            # if st.buttn("Dowload ss"):
                                            #     myScreenshot.save(r'name.png')   
                                            # myScreenshot.save(r'/home/anas/PersonTracking/AIComputerVision-master/pages/name.png')
                                    if obj == 67:
                                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                                        rectx1,recty1 = ((x1+x2)/2,(y1+y2)/2)
                                        rectcenter = int(rectx1),int(recty1)
                                        cx = rectcenter[0]
                                        cy = rectcenter[1]
                                        cv2.circle(frame,(cx,cy),3,(0,255,0),-1)
                                        cv2.putText(frame,str(mobile), (x1,y1), cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                        cv2.putText(frame,'Mobile',(50, 50),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2,cv2.LINE_4)
                                        kpil3_text.write(f"<h5 style='text-align: left; color:black;'>{text}</h5>", unsafe_allow_html=True)
                                        # print(text+mobile)
                                        db["student Count"] = [lpc_txt]
                                        db['Date'] = [timez]
                                        
                                        db['id'] = [text]
                                        db['Mobile']=mobile+' '+text
                                        name = "screenshot/"+str(timez) + '.jpg'
                                        db['Image Path']=name
                                        print(name)
                                        # print ('Creating...' + name)
                                
                                        print(timez)
                                        # writing the extracted images
                                        cv2.imwrite(name, frame)
    
                                        # myScreenshot = pyautogui.screenshot()
                                        # if st.buttn("Dowload ss"):
                                        # myScreenshot.save(r'/home/anas/PersonTracking/AIComputerVision-master/pages/name.png')
                                        # myScreenshot.save(r'name.png')     
                                 # watch
                                    if obj == 75:
                                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                                        rectx1,recty1 = ((x1+x2)/2,(y1+y2)/2)
                                        rectcenter = int(rectx1),int(recty1)
                                        cx = rectcenter[0]
                                        cy = rectcenter[1]
                                        cv2.circle(frame,(cx,cy),3,(0,255,0),-1)
                                        cv2.putText(frame,str(watch), (x1,y1), cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                        cv2.putText(frame,'Watch',(50, 50),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2,cv2.LINE_4)
                                        kpil4_text.write(f"<h5 style='text-align: left; color:black;'>{watch}</h5>", unsafe_allow_html=True)
                                        db["student Count"] = [lpc_txt]
                                        db['Date'] = [timez]
                                        
                                        db['id'] = [text]
                                        
                                        db['Mobile']=watch
                                        name = "screenshot/"+str(timez) + '.jpg'
                                        print ('Creating...' + name)
                                        db['Image Path']=name
                                        cv2.imwrite(name, frame)
                                        print(timez)
                                        # writing the extracted images
                                        
                                        # myScreenshot = pyautogui.screenshot()
                                        # if st.buttn("Dowload ss"):
                                                # myScreenshot.save(r'/home/anas/PersonTracking/AIComputerVision-master/pages/name.png')
                                        # myScreenshot.save(r'name.png')  
                                    
                                        
                                
                                # kpil_text.write(f"<h5 style='text-align: left; color:red;'>{int(fps)}</h5>", unsafe_allow_html=True)
                                kpil5_text.write(f"<h5 style='text-align: left; color:black;'>{lpc_txt}</h5>", unsafe_allow_html=True)
                                # kpil6_text.write(f"<h5 style='text-align: left; color:red;'>{width*height}</h5>",
                                                # unsafe_allow_html=True)
                                
                
                                frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
                                frame = image_resize(image=frame, width=640)
                                stframe.image(frame,channels='BGR', use_column_width=True)
                                df = pd.DataFrame(db)
                                df.to_csv('final.csv',mode='a',header=False,index=False)
                        except:
                            pass
                        with open('final.csv') as f:
                            st.download_button(label = 'Download Cheating Report',data=f,file_name='data.csv')
                            
                        os.remove("final.csv")
                main()
    else:
        st.warning("wrong username or password!")
else:
    st.write("Please login to continue!")
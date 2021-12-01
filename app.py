from flask import Flask, render_template, Response, request, make_response
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from utils import effects
from utils.record_capture import RecordVideo

dict_effect = {
    "sepia_ef":0,
    "zoom_in_ef":0,
    "time_warp_horizontal":0,
    "time_warp_vertical":0,
    "face_recognition":0,
    "vintage_ef":0,
}

convert_name = {
    "sepia_ef":"Sepia",
    "zoom_in_ef":"Zoom in",
    "time_warp_horizontal": "Time warp horizontal",
    "time_warp_vertical":"Time warp vertical",
    "face_recognition":"Face recognition",
    "vintage_ef": "Vintage"
}

choose =0
demo =0
rec=0


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


def change_values_dict(dic,key):
    keys = list(dic.keys())
    for i in keys:
        if i==key:
            dic[i]=1
        else:
            dic[i]=0

def name_effect_choosed(dic):
    for key,value in dic.items():
      if value==1:
          return convert_name[key]


def gen_frames():  
    camera = cv2.VideoCapture(0)
    global out, capture,rec_frame
    count_frame = 0
    while True:
        success, frame = camera.read() 
        count_frame +=1
        if success:
            if demo:
                if dict_effect["sepia_ef"]:
                    frame=effects.sepia_effect(frame)               
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                pass         
   
def record_frame():
    name_effect = ""
    if dict_effect["sepia_ef"]:
        name_effect = "sepia"
    elif dict_effect["zoom_in_ef"]:
        name_effect = "zoom_in"
    elif dict_effect["time_warp_horizontal"]:
        name_effect = "time_warp_scan_horizontal"
    elif dict_effect["time_warp_vertical"]:
        name_effect = "time_warp_scan_vertical"
    elif dict_effect["face_recognition"]:
        name_effect = "face_recognition"
    elif dict_effect["vintage_ef"]:
        name_effect = "vintage"
    recorder = RecordVideo(effects=name_effect)
    return recorder


    
@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    if not rec:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(record_frame().record_video_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global zoom_in_ef, sepia_ef,time_warp_horizontal, time_warp_vertical,face_recognition
    global rec, demo, choose

    if request.method == 'POST':
        if  request.form.get('effect') == 'sepia_effect':
            change_values_dict(dict_effect,"sepia_ef")

        elif  request.form.get('effect') == 'zoom_in_effect':  
            change_values_dict(dict_effect,"zoom_in_ef")

        elif  request.form.get('effect') == 'time_warp_scan_vertical':   
            change_values_dict(dict_effect,"time_warp_vertical") 

        elif  request.form.get('effect') == 'time_warp_scan_horizontal':   
            change_values_dict(dict_effect,"time_warp_horizontal")  

        elif  request.form.get('effect') == 'face_recognition':     
            change_values_dict(dict_effect,"face_recognition")     

        elif  request.form.get('effect') == 'vintage_effect':     
            change_values_dict(dict_effect,"vintage_ef")   


        elif  request.form.get('demo') == 'Demo':
            global demo
            demo = not demo
        elif  request.form.get('rec') == 'Rec':
            global rec
            rec = not rec
            
      
 
    elif request.method=='GET':
        return render_template("index.html",effedt_choosed = name_effect_choosed(dict_effect))
    return render_template("index.html",effedt_choosed = name_effect_choosed(dict_effect))


if __name__ == '__main__':
    app.run(debug=True)
    

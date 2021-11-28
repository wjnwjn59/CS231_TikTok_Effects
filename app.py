from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from utils import effects
from utils.record_capture import RecordVideo

global capture,rec_frame, grey, switch, neg, slow_zoom_ef, rec, out 
capture=0
grey=0
sepia_ef=0
slow_zoom_ef=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')




# def record(out):
#     global rec_frame
#     while(rec):
#         #time.sleep(0.05)
#         out.write(rec_frame)


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    global out, capture,rec_frame
    count_frame = 0
    while True:
        success, frame = camera.read() 
        count_frame +=1
        if success:
            
            if(slow_zoom_ef):
    

                frame =effects.zoom_in_effect(frame,count_frame,stop_zoom=50, smooth=5)


            elif(grey):   
             
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            elif(sepia_ef):
         
                frame=effects.sepia_effect(frame)

            
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass         
        else:
            pass
    camera.release()

name  = "face_recognition"
recorder = RecordVideo(effects=name)

    


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    if not rec:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(recorder.record_video_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')
 




@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
        
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('sepia_ef') == 'sepia_effect':
            global sepia_ef
            sepia_ef=not sepia_ef
        elif  request.form.get('zoom_in_ef') == 'zoom_in_effect':
            global slow_zoom_ef
            slow_zoom_ef=not slow_zoom_ef

        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            #if(rec):
                #now=datetime.datetime.now() 
                #fourcc = cv2.VideoWriter_fourcc(*'XVID')
                #out = cv2.VideoWriter('videos/vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                #thread = Thread(target = record, args=[out,])
                #thread.start()
                #if(rec==False):
                #    out.release()
             
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
#camera.release()
#cv2.destroyAllWindows()     
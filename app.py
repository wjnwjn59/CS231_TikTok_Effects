from flask import Flask,render_template,Response
import cv2
from flask.wrappers import Request
from utils import slow_zoom

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    count_frame = 0
    while True:
        
        ## read the camera frame
        check,frame=camera.read()
        count_frame +=1
        if not check:
            break
        else:
            
            frame = cv2.flip(frame,1)
            frame =slow_zoom.zoom_in(frame,count_frame)
            ret,buffer=cv2.imencode('.jpg',frame)
            
            frame=buffer.tobytes()
            
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/',methods =["GET","POST"])
def index():
    if Request.method == "POST":
        if Request.form.get("effect_1") =="value_1":
            pass
    elif Request.method == "GET":
        return render_template('index.html')
    return render_template('index.html')



@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)


    #<img src="{{ url_for('video') }}" width="100%"/>
import cv2
import time
import sys
import os
import numpy as np
import utils.effects

class RecordVideo(object):
    def __init__(self, 
                record_directory_name="recorded_videos", 
                record_name="recorded_video.mp4", 
                time_capture=10, 
                import_pic_path="import_pics/dh_cntt.jpg", 
                import_video_path=None, 
                effects=None):

        self.record_directory_name = record_directory_name
        self.record_name = record_name
        self.time_capture = time_capture
        self.import_pic_path = import_pic_path
        self.import_video_path = import_video_path
        self.effects = effects
        self.record_screen_shape = (640, 480) # (width, height)

    def record_video_capture(self):
        vid = cv2.VideoCapture(0)
        vid1=cv2.VideoCapture('import_pics/Green Screen_2.mp4')
        if not os.path.isdir(self.record_directory_name):
            os.mkdir(self.record_directory_name)
        video_name = os.path.join(self.record_directory_name, self.record_name)
        save_vid = cv2.VideoWriter(video_name, -1, 20.0, self.record_screen_shape)
        start_time = time.time()
        

        if self.effects == "background_removal":
            while (vid.isOpened()):
                ret, frame = vid.read()

                if ret:
                    effect_frame = utils.effects.background_removal_effect(frame, self.import_pic_path)
                    save_vid.write(effect_frame)

                    # cv2.imshow("frame", effect_frame)

                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(effect_frame,1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
 
        elif self.effects == "zoom_in":
            frame_count = 0
            stop_zoom = 100
            smooth = 5
            while (vid.isOpened()):
                ret, frame = vid.read()
                if ret:
                    effect_frame = utils.effects.zoom_in_effect(frame, frame_count, stop_zoom, smooth)
                    frame_count += 1
                    save_vid.write(effect_frame)

                    # cv2.imshow("frame", effect_frame)
                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(effect_frame,1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
        
        elif self.effects == "sepia":
            while (vid.isOpened()):
                ret, frame = vid.read()
                if ret:
                    effect_frame = utils.effects.sepia_effect(frame)
                    save_vid.write(effect_frame)
                    # cv2.imshow("frame", effect_frame)
                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(effect_frame,1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
        
        elif self.effects == "vintage":
            while (vid.isOpened()):
                ret, frame = vid1.read()  
                ret_1,frame_1=vid.read() 
                if ret==True:
                    frame=cv2.resize(frame,(640,480))
                    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                    l_green=np.array([32,94,132])
                    u_green=np.array([179,255,255])
                    mask=cv2.inRange(hsv,l_green,u_green)
                    res=cv2.bitwise_and(frame,frame,mask=mask)
                    f = frame-res
                    green_screen=np.where(f==0,frame_1,f)
                    # cv2.imshow('green',green_screen)
                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(green_screen, 1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
        elif self.effects == "time_warp_scan_horizontal":
            i = 0
            previous_frame = np.zeros((self.record_screen_shape[1], self.record_screen_shape[0], 3), dtype="uint8")
            cyan_line = np.zeros((self.record_screen_shape[1], 1, 3), dtype="uint8")
            cyan_line[:] = (255, 255, 0)
            while (vid.isOpened() and i < self.record_screen_shape[0]):
                ret, frame = vid.read()
                if ret:
                    previous_frame[:, i, :] = frame[:, i, :]
                    effect_frame = np.hstack((previous_frame[:, :i, :], cyan_line, frame[:, i+1:, :]))
                    save_vid.write(effect_frame)
                    #cv2.imshow("frame", effect_frame)
                    i += 1
                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(effect_frame, 1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
        elif self.effects == "time_warp_scan_vertical":
            i=0
            previous_frame_vertical=np.zeros((self.record_screen_shape[1], self.record_screen_shape[0], 3), dtype="uint8")
            cyan_line_vertical=np.zeros((1,self.record_screen_shape[0], 3), dtype="uint8")
            cyan_line_vertical[:,:] = (255, 255, 0)
            while (vid.isOpened() and i < self.record_screen_shape[1]):
                ret, frame = vid.read()
                if ret:
                    previous_frame_vertical[i, :, :] = frame[i, :, :]
                    effect_frame = np.vstack((previous_frame_vertical[:i,:, :], cyan_line_vertical, frame[i+1:,:, :]))
                    save_vid.write(effect_frame)
                    #cv2.imshow("frame", effect_frame)
                    i += 1
                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(effect_frame, 1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
        elif self.effects == "face_recognition":
            cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
            cascPath = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
            faceCascade = cv2.CascadeClassifier(cascPath)
            while (vid.isOpened()):
                 # Capture frame-by-frame
                ret, frame = vid.read()

                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = faceCascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.05,
                        minNeighbors=7, 
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # Draw a rectangle around the faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    save_vid.write(frame)
                    # Display the resulting frame
                    # cv2.imshow("frame", frame)

                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   
        else:
            while (vid.isOpened()):
                ret, frame = vid.read()

                if ret:
                    save_vid.write(frame)

                    # cv2.imshow("frame", frame)
                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   

        vid.release()
        save_vid.release()
        cv2.destroyAllWindows()



import cv2
import time
from utils import slow_zoom

def Record_Video_Capture(time_capture):
    video = cv2.VideoCapture(0)
    #name_video = "save_video1.mp4"
    #save_vid = cv2.VideoWriter(name_video,-1,20.0,(640,480))
    count_frame = 0
    start_time = time.time()
    while (int(time.time()-start_time)<time_capture):
        check,frame = video.read()
        count_frame +=1
        if check:
            #save_vid.write(frame)
            #frame=slow_zoom.zoom(frame,2)

            # lật frame lại vì bị ngược khi sử dụng webcam
            frame = cv2.flip(frame,1)
            
         
            # Module----------------------------
            
            frame =slow_zoom.zoom_in(frame,count_frame)
            
            #-----------------------------------
            
            cv2.imshow("frame", frame)
            #cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            print("Miss frame !")
            break

    video.release()
    #save_vid.release()
    cv2.destroyAllWindows()
    return count_frame

count= Record_Video_Capture(10)

print(count)
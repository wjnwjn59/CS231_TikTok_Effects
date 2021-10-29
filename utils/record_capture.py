import cv2
import time

def Record_Video_Capture(time_capture):
    vid = cv2.VideoCapture(0)
    name_video = "save_video.mp4"
    save_vid = cv2.VideoWriter(name_video,-1,20.0,(640,480))

    time_cap = time_capture
    start_time = time.time()
    while (int(time.time()-start_time)<time_capture):
        ret,frame = vid.read()

        if ret:
            save_vid.write(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    vid.release()
    save_vid.release()
    cv2.destroyAllWindows()



Record_Video_Capture(10)


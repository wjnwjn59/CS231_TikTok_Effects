import cv2
import time
import os

def Record_Video_Capture(time_capture):
    vid = cv2.VideoCapture(0)
    recorded_video_folder_name = "recorded_videos"
    if not os.path.isdir(recorded_video_folder_name):
        os.mkdir(recorded_video_folder_name)
    video_name = os.path.join(recorded_video_folder_name, "save_video.mp4")
    save_vid = cv2.VideoWriter(video_name, -1, 20.0, (640,480))
    time_cap = time_capture
    start_time = time.time()
    while (int(time.time() - start_time) < time_capture):
        ret, frame = vid.read()

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



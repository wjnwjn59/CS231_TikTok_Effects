import cv2
import time
import os
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

    def record_video_capture(self):
        vid = cv2.VideoCapture(0)
        if not os.path.isdir(self.record_directory_name):
            os.mkdir(self.record_directory_name)
        video_name = os.path.join(self.record_directory_name, self.record_name)
        save_vid = cv2.VideoWriter(video_name, -1, 20.0, (640,480))
        start_time = time.time()
        

        if self.effects == "background_removal":
            while (vid.isOpened()):
                ret, frame = vid.read()

                if ret:
                    effect_frame = utils.effects.background_removal_effect(frame, self.import_pic_path)
                    save_vid.write(effect_frame)

                    cv2.imshow("frame", effect_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
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

                    cv2.imshow("frame", effect_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
        else:
            while (vid.isOpened()):
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



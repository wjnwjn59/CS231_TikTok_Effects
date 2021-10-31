import cv2
import pixellib
from pixellib.tune_bg import alter_bg

def do_nothing_effect(frame):
    """
    If no effect is specified, then this function will be used because it returns original frame.
    """
    return frame

def background_removal_effect(frame, background_img_path):
    """
    Change background with an imported picture
    """

    change_bg = alter_bg(model_type = "pb")
    model_path = "models/pixellib_models/xception_pascalvoc.pb"
    change_bg.load_pascalvoc_model(model_path)
    output_img = change_bg.change_frame_bg(frame, background_img_path, detect="person")
    #change_bg.change_camera_bg(vid, background_img_path, frames_per_second = 60, show_frames=True, frame_name="frame", output_video_name="output_video.mp4", detect = "person")

    return output_img



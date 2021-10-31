import cv2
import pixellib
from pixellib.tune_bg import alter_bg

def do_nothing_effect(frame):

    return frame

def background_removal_effect(frame, background_img):
    """
    Change background with an imported picture
    """

    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    output_img = change_bg.change_bg_img(frame, background_img, output_image_name="new_img.jpg")

    return output_img



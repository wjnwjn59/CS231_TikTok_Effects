import cv2
import numpy as np 
# import pixellib
# from pixellib.tune_bg import alter_bg

# def background_removal_effect(frame, background_img_path):
#     """
#     Change background with an imported picture
#     """

#     change_bg = alter_bg(model_type = "pb")
#     model_path = "models/pixellib_models/xception_pascalvoc.pb"
#     change_bg.load_pascalvoc_model(model_path)
#     output_img = change_bg.change_frame_bg(frame, background_img_path, detect="person")
#     #change_bg.change_camera_bg(vid, background_img_path, frames_per_second = 60, show_frames=True, frame_name="frame", output_video_name="output_video.mp4", detect = "person")

#     return output_img


def zoom_in_effect(img, count_frame, stop_zoom=50, smooth=5):
    '''
    Input (type: np.array): Một frame ảnh.
    Ouput (type: np.array): Một frame ảnh sau khi biến đổi .
    Note:
    - count_frame (type: int): Số lượng frame đã được đọc cho tới thời điểm hiện tại.
    - stop_zoom (type: int) : khi đạt đến frame này (VD: frame thứ 50) thì dừng việc zoom lại
                                và cố định mức độ zoom trước đó.
    - smooth (type: int): smooth càng lớn thì phần crop ảnh cho mỗi frame càng ít.
                            Giá trị smooth cần điều chỉnh phù hợp với fps đọc được.
    '''
    
    if count_frame >= stop_zoom:

        scale =int(stop_zoom/smooth)
    else:
        scale = int(count_frame/smooth)
    
    height, width, _ = img.shape

    radiusX,radiusY= int(scale*height/100),int(scale*width/100)

    minX,maxX=radiusX,height-radiusX
    minY,maxY=radiusY,width-radiusY

    cropped = img[minX:maxX, minY:maxY]
    resized_cropped  = cv2.resize(cropped, (width, height)) 

    return resized_cropped

def sepia_effect(img):
    '''
    Input : Numpy Array 
    Ouput : Numpy Array
    ---------------------------------
    Function : Read numpy array,multipying with a kernel to get a special effect on result image
    '''
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    
    return img_sepia



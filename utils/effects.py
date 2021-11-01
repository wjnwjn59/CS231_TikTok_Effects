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


def zoom_in(img,count_frame):
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
    stop_zoom = 50
    smooth = 5
    
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
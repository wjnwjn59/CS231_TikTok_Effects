import cv2
import numpy as np


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




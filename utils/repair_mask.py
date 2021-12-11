import cv2
import itertools
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                            min_detection_confidence=0.5)

face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                            min_detection_confidence=0.5,min_tracking_confidence=0.3)

mp_drawing_styles = mp.solutions.drawing_styles


def detectFacialLandmarks(image, face_mesh):

    results = face_mesh.process(image[:,:,::-1])  
    output_image = image[:,:,::-1].copy()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None, 
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None, 
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    return np.ascontiguousarray(output_image[:,:,::-1], dtype=np.uint8), results    


def getSize(image, face_landmarks, INDEXES):

    image_height, image_width, _ = image.shape
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []

    for INDEX in INDEXES_LIST:
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                                int(face_landmarks.landmark[INDEX].y * image_height)])
                                
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    landmarks = np.array(landmarks)
    return width, height, landmarks


def isOpen(image, face_mesh_results, face_part, threshold=5):
    image_height, image_width, _ = image.shape
    output_image = image.copy()
    status={}

    if face_part == 'MOUTH':
        INDEXES = mp_face_mesh.FACEMESH_LIPS
        loc = (10, image_height - image_height//40)
        increment=-30

    elif face_part == 'LEFT EYE':
        INDEXES = mp_face_mesh.FACEMESH_LEFT_EYE
        loc = (10, 30)
        increment=30

    elif face_part == 'RIGHT EYE':
        INDEXES = mp_face_mesh.FACEMESH_RIGHT_EYE 
        loc = (image_width-300, 30)
        increment=30

    else:
        return

    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

        _, height, _ = getSize(image, face_landmarks, INDEXES)

        _, face_height, _ = getSize(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)

        if (height/face_height)*100 > threshold:
            status[face_no] = 'OPEN'
            color=(0,255,0)
        else:
            status[face_no] = 'CLOSE'
            color=(0,0,255)

        cv2.putText(output_image, f'FACE {face_no+1} {face_part} {status[face_no]}.', 
                    (loc[0],loc[1]+(face_no*increment)), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)

    return output_image, status


def overlay(image, filter_img, face_landmarks, face_part, INDEXES):

    annotated_image = image.copy()
    try:

        filter_img_height, filter_img_width, _  = filter_img.shape

        _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)

        required_height = int(face_part_height*2.5)

        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width*
                                                            (required_height/filter_img_height)),
                                                        required_height))

        filter_img_height, filter_img_width, _  = resized_filter_img.shape

        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                            25, 255, cv2.THRESH_BINARY_INV)

        center = landmarks.mean(axis=0).astype("int")

        if face_part == 'MOUTH':
            location = (int(center[0] - filter_img_width / 3), int(center[1]))

        else:
            location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))

        ROI = image[location[1]: location[1] + filter_img_height,
                    location[0]: location[0] + filter_img_width]

        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
        resultant_image = cv2.add(resultant_image, resized_filter_img)

        annotated_image[location[1]: location[1] + filter_img_height,
                        location[0]: location[0] + filter_img_width] = resultant_image

    except Exception as e:
        pass

    return annotated_image
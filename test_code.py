# import the necessary packages
from imutils import face_utils
import dlib
import cv2
from utils import repair_mask
import time
import numpy as np
# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)



# Create named window for resizing purposes.
cv2.namedWindow('Face Landmarks Detection', cv2.WINDOW_NORMAL)

def detectFacialLandmarks(image, face_mesh):

    results = face_mesh.process(image[:,:,::-1])  
    output_image = image[:,:,::-1].copy()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            repair_mask.mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                            connections=repair_mask.mp_face_mesh.FACEMESH_TESSELATION,
                                            landmark_drawing_spec=None, 
                                            connection_drawing_spec=repair_mask.mp_drawing_styles.get_default_face_mesh_tesselation_style())

            repair_mask.mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                        connections=repair_mask.mp_face_mesh.FACEMESH_LEFT_EYE,
                                        landmark_drawing_spec=None, 
                                        connection_drawing_spec=repair_mask.mp_drawing_styles.get_default_face_mesh_contours_style())
            
            repair_mask.mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                        connections=repair_mask.mp_face_mesh.FACEMESH_RIGHT_EYE,
                                        landmark_drawing_spec=None, 
                                        connection_drawing_spec=repair_mask.mp_drawing_styles.get_default_face_mesh_contours_style())
            repair_mask.mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                        connections=repair_mask.mp_face_mesh.FACEMESH_LIPS,
                                        landmark_drawing_spec=None, 
                                        connection_drawing_spec=repair_mask.mp_drawing_styles.get_default_face_mesh_contours_style())

            repair_mask.mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                        connections=repair_mask.mp_face_mesh.FACEMESH_FACE_OVAL,
                                        landmark_drawing_spec=None, 
                                        connection_drawing_spec=repair_mask.mp_drawing_styles.get_default_face_mesh_contours_style())

    return np.ascontiguousarray(output_image[:,:,::-1], dtype=np.uint8), results    




# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to 
    # read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Perform Face landmarks detection.
    frame, face_mesh_results = detectFacialLandmarks(frame, repair_mask.face_mesh_videos)
    mouth_status = dict()
    left_eye_status= dict()
    right_eye_status= dict()
    if face_mesh_results.multi_face_landmarks:
        _, mouth_status = repair_mask.isOpen(frame, face_mesh_results, 'MOUTH', threshold=15)
        _, left_eye_status = repair_mask.isOpen(frame, face_mesh_results, 'LEFT EYE', threshold=5)
        _, right_eye_status = repair_mask.isOpen(frame, face_mesh_results, 'RIGHT EYE', threshold=5)
  

    
    if len(left_eye_status)!=0:
        cv2.putText(frame, 'Left eye: {}'.format(left_eye_status[0]), (500, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    if len(right_eye_status)!=0:
        cv2.putText(frame, 'Right eye: {}'.format(right_eye_status[0]), (500,130),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    if len(mouth_status)!=0:
        cv2.putText(frame, 'Mouth: {}'.format(mouth_status[0]), (500, 160),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)


    cv2.imshow('Face Landmarks Detection', frame)
    
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF    
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.                  
camera_video.release()
cv2.destroyAllWindows()
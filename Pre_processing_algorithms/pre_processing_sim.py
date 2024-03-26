import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy.spatial import distance
import os
from insightface.utils import face_align
from skimage import transform as trans

# Training/validation input and output directories
input_dir = '../VGG'
output_dir = '../Img_processed_train/VGG_preproc_sim'

# Input and output directories for testing
# input_dir = '../lfw-deepfunneled'
# output_dir = '../Img_processed_test/LFW_preproc_sim'

#Size of the images for the training set
desiredFaceWidth = 144
desiredFaceHeight = 144 
            
#Size of the images for the test set
# desiredFaceWidth = 128
# desiredFaceHeight = 128

# Create the output folder if it doesn't already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
app.prepare(ctx_id=0, det_size=(640, 640))

# Loop through all the subfolders in the "lfw" folder
for person_name in os.listdir(input_dir):
    person_dir = os.path.join(input_dir, person_name)

    # Skip if the item in the "lfw" folder is not a folder
    if not os.path.isdir(person_dir):
        continue

    # Create a new folder with the same name in the "lfw_preprocessed" folder
    person_output_dir = os.path.join(output_dir, person_name)
    if not os.path.exists(person_output_dir):
        os.makedirs(person_output_dir)

    # Loop through all the images in the current folder
    for image_name in os.listdir(person_dir):
        # Ignore files that are not images
        if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
            continue

        # Load the image and detect the landmarks
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)
        faces = app.get(img)

        if len(faces) == 0:
            # Skip if there is no face detected in the image
            continue
        else:
            face_bbox = faces[0]['bbox']
            face = img[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
            landmarks = faces[0]['landmark_2d_106']

                
            # Always have to put in an extra point if I don't count the last point.
            # Select the landmarks of interest
            eye_left = landmarks[33:43]
            eye_right = landmarks[87:97]
            nose = landmarks[72:87]
            mouth = landmarks[52:72]


            # Desired eye position and desired final image size
            desiredLeftEye = (0.35, 0.35)            
            desiredRightEyeX = 1.0 - desiredLeftEye[0]

            # Calculate the centre of each eye
            eye_left_points = np.concatenate([eye_left], axis=0)
            eye_right_points = np.concatenate([ eye_right], axis=0)
                        
            eye_left_center = np.mean(eye_left_points, axis=0)
            eye_right_center = np.mean(eye_right_points, axis=0)

            # Eye centre
            eyesCenter = ((eye_left_center[0] + eye_right_center[0]) // 2, (eye_left_center[1] + eye_right_center[1]) // 2)

            # Draw circles at the eye coordinates
            # cv2.circle(img, tuple(map(int, eye_left_center)), 5, (0, 0, 255), -1)
            # cv2.circle(img, tuple(map(int, eye_right_center)), 5, (0, 0, 255), -1)

            # Calculate the angle of rotation

            # Position of the centre of each eye
            dY = eye_right_center[1] - eye_left_center[1]
            dX = eye_right_center[0] - eye_left_center[0]
            # Calculate the angle between the line connecting the eye points and the x-axis
            angle = np.degrees(np.arctan2(dY, dX))

            # Calculate the scale
            # Distance between the centre of the two eyes
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            # Distance the eyes will have from each other
            desiredDist = (desiredRightEyeX - desiredLeftEye[0])
            desiredDist *= desiredFaceWidth
            # Desired scale
            scale = desiredDist / dist

            rot = float(-angle) * np.pi / 180.0
            # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
            t1 = trans.SimilarityTransform(scale=scale)
            t2 = trans.SimilarityTransform(translation=(-eyesCenter[0]*scale, -eyesCenter[1]*scale))
            t3 = trans.SimilarityTransform(rotation=rot)
            t4 = trans.SimilarityTransform(translation=(desiredFaceWidth / 2,
                                                (desiredFaceHeight)* desiredLeftEye[1]))
            t = t1 + t2 + t3 + t4
            M = t.params[0:2]

            aimg = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight),borderValue=0.0)
            gray_img = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY)
            # Save the preprocessed image in the person's folder
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, aimg)

from mtcnn import MTCNN
import cv2
import numpy as np
from skimage import transform as trans
import os

# Training/validation input and output directories
input_dir = '../VGG'
output_dir = '../Img_processed_train/VGG_preproc_mtcnn'

# Input and output directories for testing
# input_dir = '../lfw-deepfunneled'
# output_dir = '../Img_processed_test/LFW_preproc_mtcnn'

#  Size of the images for the training set
desiredFaceWidth = 144
desiredFaceHeight = 144 

# Size of the images for the test set
# desiredFaceWidth = 128
# desiredFaceHeight = 128

# Create the output folder if it doesn't already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        
        image_path = os.path.join(person_dir, image_name)
        # Loading the input image
        img = cv2.imread(image_path)

        # Initialising the MTCNN detector
        detector = MTCNN()

        # Detecting faces in the image
        results = detector.detect_faces(img)
        
        if len(results) == 0:
            # Skip if there is no face detected in the image
            continue
        else:

            # Drawing a rectangle around each detected face

            x, y, w, h = results[0]['box']
            keypoints = results[0]['keypoints']
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            left_eye = results[0]['keypoints']['left_eye']
            right_eye = results[0]['keypoints']['right_eye']
            nose = results[0]['keypoints']['nose']
            mouth_left = results[0]['keypoints']['mouth_left']
            mouth_right = results[0]['keypoints']['mouth_right']
            
            for _, values in results[0]['keypoints'].items():
                cv2.circle(img, values, 3, (0, 255, 0), -1)
            # cv2.circle(img, mouth_left, 2, (0, 0, 255), -1)

            # Desired eye position
            desiredLeftEye = (0.35, 0.35)        
            desiredRightEyeX = 1.0 - desiredLeftEye[0]
             
            # Eye centre
            eyesCenter = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            # Position of the centre of each eye
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
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

            # Rotation matrix
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            # Translation matrix
            tx = desiredFaceWidth * 0.5
            ty = desiredFaceHeight * desiredLeftEye[1]
            M[0, 2] += (tx - eyesCenter[0])
            M[1, 2] += (ty - eyesCenter[1])

            # Affinity transformation
            aligned = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)

            gray_img = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, aligned)
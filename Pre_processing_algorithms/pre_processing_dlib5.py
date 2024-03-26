import dlib
import cv2
import numpy as np
from skimage import transform as trans
import os

# Training/validation input and output directories
input_dir = '../VGG'
output_dir = '../Img_processed_train/VGG_preproc_dlib5'

# Input and output directories for testing
# input_dir = '../lfw-deepfunneled'
# output_dir = '../Img_processed_test/LFW_preproc_dlib5'

# Size of the images for the training set
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
        # Ignora arquivos que não são imagens
        if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
            continue

        # Loading the pre-trained model
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)

        # Detecting faces in the image
        faces = detector(img)

        if len(faces) == 0:
            # Skip if there is no face detected in the image
            continue
        else:

            landmarks = predictor(img, faces[0])

            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)

            olho_dir_dir_x = landmarks[0][0]
            olho_dir_dir_y = landmarks[0][1]

            olho_dir_esq_x = landmarks[1][0]
            olho_dir_esq_y = landmarks[1][1]

            olho_esq_esq_x = landmarks[2][0]
            olho_esq_esq_y = landmarks[2][1]

            olho_esq_dir_x = landmarks[3][0]
            olho_esq_dir_y = landmarks[3][1]

            nariz_x = landmarks[4][0]
            nariz_y = landmarks[4][1]


            # Desired eye position
            desiredLeftEye = (0.35, 0.35)           
            desiredRightEyeX = 1.0 - desiredLeftEye[0]

            # Drawing the dots on the original image
            # for (x, y) in landmarks:
            # cv2.circle(img, (olho_dir_dir_x, olho_dir_dir_y), 1, (0, 0, 255), -1)

            # eye_left_points = np.concatenate([eye_left], axis=0)
            eye_right_points = np.concatenate(([landmarks[0]],[landmarks[1]]), axis=0)
            eye_left_points = np.concatenate(([landmarks[2]],[landmarks[3]]), axis=0)


            eye_right_center = np.mean(eye_right_points, axis=0)
            eye_left_center = np.mean(eye_left_points, axis=0)

            # Eye centre
            eyesCenter = ((eye_left_center[0] + eye_right_center[0]) // 2, (eye_left_center[1] + eye_right_center[1]) // 2)

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

            angle = np.degrees(np.arctan2(dY, dX))

            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            # Translation matrix
            tx = desiredFaceWidth * 0.5
            ty = desiredFaceHeight * desiredLeftEye[1]
            M[0, 2] += (tx - eyesCenter[0])
            M[1, 2] += (ty - eyesCenter[1])

            # Affinity transformation
            aligned = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)

            #cv2.circle(img, (eye_right_center[0], eye_right_center[1]), 1, (0, 0, 255), -1)   
            #cv2.circle(img, (eye_left_center[0], eye_left_center[1]), 1, (0, 0, 255), -1)   
            gray_img = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, gray_img)


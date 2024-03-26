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
output_dir = '../Img_processed_train/VGG_preproc_proj_2'

# Input and output directories for testing
# input_dir = '../lfw-deepfunneled'
# output_dir = '../Img_processed_test/LFW_preproc_proj_2'

#Size of the images for the training set
desiredFaceWidth = 144
desiredFaceHeight = 144 
            
#Size of the images for the test set
#desiredFaceWidth = 128
#desiredFaceHeight = 128

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
        # Ignora arquivos que não são imagens
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

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Destination points
            dst = np.array([[0, 0], [desiredFaceWidth-1, 0], [0, desiredFaceHeight-1], [desiredFaceWidth-1, desiredFaceHeight-1]], dtype=np.float32)

            # Source points (corners of the cropped image)
            # Target points for the transformation (corners of the bounding box)
            bbox_points = [(face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[1]), (face_bbox[0], face_bbox[3]), (face_bbox[2], face_bbox[3])]
            src_points = np.array(bbox_points, dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_points, dst)

            dst = cv2.warpPerspective(img, M, (desiredFaceWidth, desiredFaceHeight))

            gray_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            # Save the preprocessed image in the person's folder
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, dst)

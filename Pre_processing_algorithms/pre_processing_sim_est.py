import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy.spatial import distance
import os
from skimage import transform as trans

# Training/validation input and output directories
input_dir = '../VGG'
output_dir = '../Img_processed_train/VGG_preproc_est_sim'

# Input and output directories for testing
# input_dir = '../lfw-deepfunneled'
# output_dir = '../Img_processed_test/LFW_preproc_est_sim'

# Size of the images for the training set
desiredFaceWidth = 144
desiredFaceHeight = 144 
            
# Size of the images for the test set
# desiredFaceWidth = 128
# desiredFaceHeight = 128

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%144==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/144.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

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

            left_eye = landmarks[38:39]
            right_eye = landmarks[88:89]
            left_mouth = landmarks[52:53]
            right_mouth = landmarks[61:62]
            center_nose = landmarks[86:87]

            landmarks_orig = np.array([[left_eye[0][0], left_eye[0][1]], [right_eye[0][0], right_eye[0][1]], [center_nose[0][0], center_nose[0][1]], [left_mouth[0][0], left_mouth[0][1]], [right_mouth[0][0], right_mouth[0][1]]], dtype=np.float32)
            aimg = norm_crop(img, landmarks_orig, image_size=desiredFaceWidth, mode='arcface')

            gray_img = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY)
            # Save the preprocessed image in the person's folder
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, gray_img)
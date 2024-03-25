import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy.spatial import distance
import os
from insightface.utils import face_align
from skimage import transform as trans

#Diretórios de entrada e saída treino/validação
input_dir = '../VGG'
output_dir = '../Img_processadas_treino/VGG_preproc_proj'

#Diretórios de entrada e saída para teste
#input_dir = '../lfw-deepfunneled'
#output_dir = '../Img_processadas_teste/LFW_preproc_proj'

#Tamanho das imagens para o conjunto de treino
desiredFaceWidth = 144
desiredFaceHeight = 144 
            
#Tamanho das imagens para o conjunto de teste
#desiredFaceWidth = 128
#desiredFaceHeight = 128

#Cria a pasta de saída se ela ainda não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
app.prepare(ctx_id=0, det_size=(640, 640))

# Loop através de todas as subpastas na pasta "lfw"
for person_name in os.listdir(input_dir):
    person_dir = os.path.join(input_dir, person_name)

    # Pula se o item na pasta "lfw" não for uma pasta
    if not os.path.isdir(person_dir):
        continue

    # Cria uma nova pasta com o mesmo nome na pasta "lfw_preprocessadas"
    person_output_dir = os.path.join(output_dir, person_name)
    if not os.path.exists(person_output_dir):
        os.makedirs(person_output_dir)

    # Loop através de todas as imagens na pasta atual
    for image_name in os.listdir(person_dir):
        # Ignora arquivos que não são imagens
        if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
            continue

        # Carrega a imagem e detecta as landmarks
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)
        faces = app.get(img)

        if len(faces) == 0:
            # Pula se não houver nenhum rosto detectado na imagem
            continue
        else:
            face_bbox = faces[0]['bbox']
            face = img[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
            landmarks = faces[0]['landmark_2d_106']



            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            arcface_dst = np.array(
                [[38.2946+16.5, 51.6963], [73.5318+16.5, 51.5014], [41.5493+16.5, 92.3655], [70.7299+16.5, 92.2041]],
                 dtype=np.float32)
#[56.0252, 71.7366]

            left_eye = landmarks[38:39]
            right_eye = landmarks[88:89]
            left_mouth = landmarks[52:53]
            right_mouth = landmarks[61:62]
            center_nose = landmarks[86:87]

            landmarks_orig = np.array([left_eye, right_eye, left_mouth, right_mouth], dtype=np.float32) #center_nose

            M = cv2.getPerspectiveTransform(landmarks_orig, arcface_dst)

            dst = cv2.warpPerspective(img, M, (desiredFaceWidth,desiredFaceHeight))
            
            gray_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            # Guardar a imagem preprocessada na pasta da pessoa
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, gray_img)

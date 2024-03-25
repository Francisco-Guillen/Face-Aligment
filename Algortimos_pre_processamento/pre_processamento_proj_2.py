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
output_dir = '../Img_processadas_treino/VGG_preproc_proj_2'

#Diretórios de entrada e saída para teste
#input_dir = '../lfw-deepfunneled'
#output_dir = '../Img_processadas_teste/LFW_preproc_proj_2'

#Tamanho das imagens para o conjunto de treino
desiredFaceWidth = 144
desiredFaceHeight = 144 
            
#Tamanho das imagens para o conjunto de teste
#desiredFaceWidth = 128
#desiredFaceHeight = 128

# Cria a pasta de saída se ela ainda não existir
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

            # Pontos de destino
            dst = np.array([[0, 0], [desiredFaceWidth-1, 0], [0, desiredFaceHeight-1], [desiredFaceWidth-1, desiredFaceHeight-1]], dtype=np.float32)

            # Pontos de origem (cantos da imagem recortada)
            # Pontos de destino para a transformação (cantos da caixa delimitadora)
            bbox_points = [(face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[1]), (face_bbox[0], face_bbox[3]), (face_bbox[2], face_bbox[3])]
            src_points = np.array(bbox_points, dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_points, dst)

            dst = cv2.warpPerspective(img, M, (desiredFaceWidth, desiredFaceHeight))

            gray_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            # Guardar a imagem preprocessada na pasta da pessoa
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, dst)

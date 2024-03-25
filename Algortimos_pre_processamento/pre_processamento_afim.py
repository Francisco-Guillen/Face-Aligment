import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy.spatial import distance
import os


#Diretórios de entrada e saída treino/validação
input_dir = '../VGG'
output_dir = '../Img_processadas_treino/VGG_preproc_afim'

#Diretórios de entrada e saída para teste
#input_dir = '../lfw-deepfunneled'
#output_dir = '../Img_processadas_teste/LFW_preproc_afim'

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

            #Tenho sempre que meter um ponto a mais se não não conta o ultimo ponto
            # Selecionar as landmarks de interesse
            eye_left = landmarks[33:43]
            eye_right = landmarks[87:97]
            nose = landmarks[72:87]
            mouth = landmarks[52:72]
            
              
                
            #Posição dos olhos pretendida
            desiredLeftEye = (0.35, 0.35)           
            desiredRightEyeX = 1.0 - desiredLeftEye[0]

            #Calcular o centro de cada olho
            eye_left_points = np.concatenate([eye_left], axis=0)
            eye_right_points = np.concatenate([ eye_right], axis=0)
                        
            eye_left_center = np.mean(eye_left_points, axis=0)
            eye_right_center = np.mean(eye_right_points, axis=0)

            #Centro dos olhos
            eyesCenter = ((eye_left_center[0] + eye_right_center[0]) // 2, (eye_left_center[1] + eye_right_center[1]) // 2)

            #Calcular o ângulo de rotação
            #Posição do centro de cada olho
            dY = eye_right_center[1] - eye_left_center[1]
            dX = eye_right_center[0] - eye_left_center[0]
            ## Calcular o ângulo entre a linha que liga os pontos dos olhos e o eixo x
            angle = np.degrees(np.arctan2(dY, dX))

            #Calcular a escala
            #Distancia entre o centro dos dois olhos
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            #Distancia que os olhos irão ter um do outro
            desiredDist = (desiredRightEyeX - desiredLeftEye[0])
            desiredDist *= desiredFaceWidth
            #Escala desejada
            scale = desiredDist / dist

            #Matriz de rotação
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            #Matriz de translação
            tx = desiredFaceWidth * 0.5
            ty = desiredFaceHeight * desiredLeftEye[1]
            M[0, 2] += (tx - eyesCenter[0])
            M[1, 2] += (ty - eyesCenter[1])

            #Tranformação afim
            aligned = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)
            gray_img = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            
            
            # Guardar a imagem preprocessada na pasta da pessoa
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, aligned)

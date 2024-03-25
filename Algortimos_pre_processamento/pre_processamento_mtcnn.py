from mtcnn import MTCNN
import cv2
import numpy as np
from skimage import transform as trans
import os

#Diretórios de entrada e saída treino/validação
input_dir = '../VGG'
output_dir = '../Img_processadas_treino/VGG_preproc_mtcnn'

#Diretórios de entrada e saída para teste
#input_dir = '../lfw'
#output_dir = '../Img_processadas_teste/LFW_preproc_mtcnn'

#Tamanho das imagens para o conjunto de treino
desiredFaceWidth = 144
desiredFaceHeight = 144 

#Tamanho das imagens para o conjunto de teste
#desiredFaceWidth = 128
#desiredFaceHeight = 128

#Cria a pasta de saída se ela ainda não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        
        image_path = os.path.join(person_dir, image_name)
        # Carregando a imagem de entrada
        img = cv2.imread(image_path)

        # Inicializando o detector MTCNN
        detector = MTCNN()

        # Detectando rostos na imagem
        results = detector.detect_faces(img)
        
        if len(results) == 0:
            # Pula se não houver nenhum rosto detectado na imagem
            continue
        else:

            # Desenhando um retângulo em volta de cada rosto detectado

            x, y, w, h = results[0]['box']
            keypoints = results[0]['keypoints']
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            left_eye = results[0]['keypoints']['left_eye']
            right_eye = results[0]['keypoints']['right_eye']
            nose = results[0]['keypoints']['nose']
            mouth_left = results[0]['keypoints']['mouth_left']
            mouth_right = results[0]['keypoints']['mouth_right']
            
            for _, values in results[0]['keypoints'].items():
                cv2.circle(img, values, 3, (0, 255, 0), -1)
            #cv2.circle(img, mouth_left, 2, (0, 0, 255), -1)

            #Posição dos olhos pretendida
            desiredLeftEye = (0.35, 0.35)        
            desiredRightEyeX = 1.0 - desiredLeftEye[0]
            
            #Centro dos olhos
            eyesCenter = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            #Posição do centro de cada olho
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
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
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, aligned)
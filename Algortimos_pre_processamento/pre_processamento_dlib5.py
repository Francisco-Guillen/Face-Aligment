import dlib
import cv2
import numpy as np
from skimage import transform as trans
import os

#Diretórios de entrada e saída treino/validação
input_dir = '../VGG'
output_dir = '../Img_processadas_treino/VGG_preproc_dlib5'

#Diretórios de entrada e saída para teste
#input_dir = '../lfw-deepfunneled'
#output_dir = '../Img_processadas_teste/LFW_preproc_dlib5'

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

        # Carregando o modelo pré-treinado
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)

        # Detectando as faces na imagem
        faces = detector(img)

        if len(faces) == 0:
            # Pula se não houver nenhum rosto detectado na imagem
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


            #Posição dos olhos pretendida
            desiredLeftEye = (0.35, 0.35)           
            desiredRightEyeX = 1.0 - desiredLeftEye[0]

            # Desenhando os pontos na imagem original
            #for (x, y) in landmarks:
            #cv2.circle(img, (olho_dir_dir_x, olho_dir_dir_y), 1, (0, 0, 255), -1)

            #eye_left_points = np.concatenate([eye_left], axis=0)
            eye_right_points = np.concatenate(([landmarks[0]],[landmarks[1]]), axis=0)
            eye_left_points = np.concatenate(([landmarks[2]],[landmarks[3]]), axis=0)


            eye_right_center = np.mean(eye_right_points, axis=0)
            eye_left_center = np.mean(eye_left_points, axis=0)

            #Centro dos olhos
            eyesCenter = ((eye_left_center[0] + eye_right_center[0]) // 2, (eye_left_center[1] + eye_right_center[1]) // 2)

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

            angle = np.degrees(np.arctan2(dY, dX))

            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            #Matriz de translação
            tx = desiredFaceWidth * 0.5
            ty = desiredFaceHeight * desiredLeftEye[1]
            M[0, 2] += (tx - eyesCenter[0])
            M[1, 2] += (ty - eyesCenter[1])

            #Tranformação afim
            aligned = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)

            #cv2.circle(img, (eye_right_center[0], eye_right_center[1]), 1, (0, 0, 255), -1)   
            #cv2.circle(img, (eye_left_center[0], eye_left_center[1]), 1, (0, 0, 255), -1)   
            gray_img = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            preprocessed_path = os.path.join(person_output_dir, image_name)
            cv2.imwrite(preprocessed_path, gray_img)


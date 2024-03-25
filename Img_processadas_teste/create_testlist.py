import os
import glob
import argparse

def dataset_images_list(dataset_root, start_id=0):
    ds_list = []
    id = start_id
    person_ids = [os.path.basename(x) for x in glob.glob(os.path.join(dataset_root, '*'))]

    for person_id in person_ids:
        person_imgs = glob.glob(os.path.join(dataset_root, person_id, '*'))
        nimg = len(person_imgs)
        for img in person_imgs:
            img_rel_path = os.path.relpath(img, dataset_root)
            ds_list.append((img_rel_path, id))
        id = id + 1
    return ds_list

# Configurar o argumento de linha de comando
parser = argparse.ArgumentParser(description='Criação de lista de imagens e labels a partir de um diretório raiz.')
parser.add_argument('dataset_root', type=str, help='Diretório raiz do conjunto de dados')

# Obter o valor do argumento
args = parser.parse_args()

# Diretório raiz do conjunto de dados
dataset_root = args.dataset_root

# Criação da lista de imagens e labels
ds_list = dataset_images_list(dataset_root)

# Criação do arquivo de texto
dataset_imglist = open(f'TestLists/test_{dataset_root}.txt', 'w')
# Escrever os caminhos das imagens e suas labels no arquivo
for img_path, label in ds_list:
    dataset_imglist.write(f'{img_path} {label}\n')

# Fechar o arquivo
dataset_imglist.close()

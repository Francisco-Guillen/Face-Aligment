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
        if nimg == 1:
            continue

        for img in person_imgs:
            img_rel_path = os.path.relpath(img, dataset_root)
            ds_list.append((img_rel_path, id, nimg))

        id = id + 1

    return ds_list, id-1


# Configurar o argumento de linha de comando
parser = argparse.ArgumentParser(description='Criação de lista de imagens e labels a partir de um diretório raiz.')
parser.add_argument('ds4_root', type=str, help='Diretório raiz do conjunto de dados')

# Obter o valor do argumento
args = parser.parse_args()

# Diretório raiz do conjunto de dados
ds4_root = args.ds4_root

ds4_list, max_id = dataset_images_list(ds4_root, start_id=0)

print('NUM IDs: ' + str(max_id+1))

ds_list = ds4_list

from sklearn.model_selection import train_test_split
# select only images whose ID has more than 1 image
X = [ds_list[i][0] for i in range(len(ds_list)) if ds_list[i][2] > 1]
y = [ds_list[i][1] for i in range(len(ds_list)) if ds_list[i][2] > 1]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2)

dataset_imglist_train = open(f'TrainLists/train_{ds4_root}.txt', 'w')
dataset_imglist_val = open(f'ValLists/val_{ds4_root}.txt', 'w')

for x_s, y_s in zip(X_train, y_train):

    print('{0} {1}'.format(x_s, y_s), file=dataset_imglist_train)

for x_s, y_s in zip(X_test, y_test):
    print('{0} {1}'.format(x_s, y_s), file=dataset_imglist_val)


dataset_imglist_train.close()
dataset_imglist_val.close()


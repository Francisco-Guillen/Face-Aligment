import numpy as np

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label, label_name, img_att, person_att = line.strip().split(',')
            img_att = np.fromstring(img_att.lstrip('[ ').rstrip(']'), dtype=int, sep=' ')
            person_att = np.fromstring(person_att.lstrip('[ ').rstrip(']'), dtype=int, sep=' ')
            imgList.append((imgPath, int(label), label_name, img_att, person_att))
    return imgList


#fileList = '/home/socialab/Desktop/Joao/projects/LightCNN_attloss/imagelists/val_person_att.txt'
#imglist = default_list_reader(fileList)
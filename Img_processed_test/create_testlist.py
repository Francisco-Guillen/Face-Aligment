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

# Setting the command line argument
parser = argparse.ArgumentParser(description='Creating a list of images and labels from a root directory.')
parser.add_argument('dataset_root', type=str, help='Dataset root directory')

# Get the value of the argument
args = parser.parse_args()

# Dataset root directory
dataset_root = args.dataset_root

# Creating the list of images and labels
ds_list = dataset_images_list(dataset_root)

# Creating a text file
dataset_imglist = open(f'TestLists/test_{dataset_root}.txt', 'w')
# Write the paths of the images and their labels in the file
for img_path, label in ds_list:
    dataset_imglist.write(f'{img_path} {label}\n')

# Close the file
dataset_imglist.close()

from LFWPairedDataset import LFWPairedDataset

dataroot = 'Projeto/lfw-deepfunneled'
pairs_cfg = 'Projeto/lfw-dataset/pairs.txt'
#transform = ... # Defina suas transformações de imagem desejadas
batch_size = 32 # Defina o tamanho do lote desejado

dataset = LFWPairedDataset(dataroot, pairs_cfg)
print(dataset.image_names_a)
import os
from torch.utils import data
import cv2
from cv2 import imread as image_loader

class PairedFeatures(data.Dataset):

    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader

        self.image_names_a = []
        self.image_names_b = []
        self.matches = []

        self._prepare_dataset()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        return (self.loader(self.image_names_a[index]),
                self.loader(self.image_names_b[index]),
                self.matches[index])

    def _prepare_dataset(self):
        raise NotImplementedError


class LFWPairedFeatures(PairedFeatures):
    def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)

        for pair in pairs:
            if len(pair) == 3:
                match = True
                name1, name2, index1, index2 = \
                    pair[0], pair[0], int(pair[1]), int(pair[2])

            else:
                match = False
                name1, name2, index1, index2 = \
                    pair[0], pair[2], int(pair[1]), int(pair[3])

            feat_path1 = os.path.join(
                self.dataroot,
                name1, "{}_{:04d}.feat".format(name1, index1)
            )
            feat_path2 = os.path.join(
                self.dataroot,
                name2, "{}_{:04d}.feat".format(name2, index2)
            )

            # Verificar se o arquivo .feat existe para ambos os índices
            if not os.path.isfile(feat_path1) or not os.path.isfile(feat_path2):
                continue  # Pular para o próximo par de imagens

            self.image_names_a.append(feat_path1)
            self.image_names_b.append(feat_path2)
            self.matches.append(match)


    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs
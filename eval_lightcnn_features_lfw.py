import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from LightCNN.light_cnn import LightCNN_9Layers, LightCNN_29Layers_v2
from LFWPairedDataset import LFWPairedDataset
from LFWPairedFeatures import LFWPairedFeatures
#from utils.device import device
from Load_yaml import load_yaml
#from utils.general import feat_loader, load_yaml
from Load_feat import feat_loader
from lfw_evaluation import lfw_evaluation_protocol
#from utils.transforms import transform_for_infer


def evaluate_model(model, test_set_path, pairs_file, batch_size=32):
    dataset_dir = test_set_path

    eval_flip_images = False
    nrof_flips = 2 if eval_flip_images else 1

    pairs_path = pairs_file if pairs_file else \
        os.path.join(dataset_dir, 'pairs.txt')

    # if not os.path.isfile(pairs_path):
    #    download(dataset_dir, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    dataset = LFWPairedFeatures(
        dataset_dir, pairs_path, transform=None, loader=feat_loader)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    with torch.inference_mode():
        model.eval()

        embedings_a = []  # torch.zeros(len(dataset), model.FEATURE_DIM * nrof_flips)
        embedings_b = []  # torch.zeros(len(dataset), model.FEATURE_DIM * nrof_flips)
        matches = []  # torch.zeros(len(dataset), dtype=torch.uint8)

        for iteration, (feats_a, feats_b, batched_matches) \
                in enumerate(dataloader):
            current_batch_size = len(batched_matches)

            embedings_a.append(feats_a)
            embedings_b.append(feats_b)
            matches.append(batched_matches)
            print("Iteration:", iteration)
            print("Feats A:", feats_a)
            print("Feats B:", feats_b)
            print("Batched Matches:", batched_matches)
        embedings_a = torch.concat(embedings_a, axis=0)
        embedings_b = torch.concat(embedings_b, axis=0)
        matches = torch.concat(matches, axis=0)

        _acc = dict({'nol2_euc': None, 'nol2_cos': None, 'l2_euc': None, 'l2_cos': None})
        _statstest = dict({'nol2_euc': None, 'nol2_cos': None, 'l2_euc': None, 'l2_cos': None})
        _statstrain = dict({'nol2_euc': None, 'nol2_cos': None, 'l2_euc': None, 'l2_cos': None})
        _acc['nol2_euc'], _statstest['nol2_euc'], _statstrain['nol2_euc'] = \
            lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=False, dist='euclidean')
        _acc['l2_euc'], _statstest['l2_euc'], _statstrain['l2_euc'] = \
            lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=True, dist='euclidean')
        _acc['nol2_cos'], _statstest['nol2_cos'], _statstrain['nol2_cos'] = \
            lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=False, dist='cosine')
        _acc['l2_cos'], _statstest['l2_cos'], _statstrain['l2_cos'] = \
            lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=True, dist='cosine')

        print(_acc)
        del embedings_a
        del embedings_b

    return _acc, _statstest, _statstrain


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
    parser.add_argument('--test_set_path', '-t', metavar='Test Features Path', default='')

    args = parser.parse_args()

    cfg_inference = load_yaml('config.yml')
    num_classes = 5736
    #num_classes = 99478
    model = LightCNN_29Layers_v2(num_classes=num_classes)

    accuracy, stats_test, stats_train = \
        evaluate_model(model,
                       test_set_path=cfg_inference['lfw_config']['test_set_path'],
                       #test_set_path=args.test_set_path,
                       pairs_file=cfg_inference['lfw_config']['pairs_file'],
                       batch_size=cfg_inference['batch-size'])

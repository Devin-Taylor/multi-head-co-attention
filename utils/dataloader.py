import codecs
import json
import os
import pickle

import imgaug.augmenters as iaa
from imblearn.over_sampling import SMOTE  # NOTE must be install from fork https://github.com/Devin-Taylor/imbalanced-learn
import multiaug.augmenters as aug
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import dataset

from utils.ppmi_data import load_ppmi

ROOT = "./"
DATA = os.path.join(ROOT, "data/")

np.random.seed(1477)

class PPMIDataset(dataset.Dataset):
    def __init__(self, train, classification=False, balance=False, augmentation=False, fraction=0.2, max_angle=5, fold=None, folds=5):
        self.seed = 1447
        self.classification = classification

        if fold is None: # don't use kfolds
            with open(os.path.join(DATA, "ppmi_all_train_test_indices_unbalanced.json")) as fd:
                data = json.load(fd)
        else:
            with open(os.path.join(DATA, "ppmi_all_train_test_indices_folds.json")) as fd:
                data = json.load(fd)


        cont_meth, cont_spect, _, cont_meta = load_ppmi(dataset="CONTROL", normalise=True)
        pd_meth, pd_spect, _, pd_meta = load_ppmi(dataset="PD", normalise=True)

        if train:
            if fold is None:
                cont_train_ids = data['control']['train_unbalanced']
                pd_train_ids = data['pd']['train_unbalanced']
            else:
                train_folds = [x for x in range(folds) if x != fold]
                cont_train_ids = []
                pd_train_ids = []
                for f in train_folds:
                    cont_train_ids += data['control']['folds'][str(f)]
                    pd_train_ids += data['pd']['folds'][str(f)]

            cont_meth = cont_meth.iloc[cont_train_ids, :]
            pd_meth = pd_meth.iloc[pd_train_ids, :]
            cont_meta = cont_meta.iloc[cont_train_ids, :]
            pd_meta = pd_meta.iloc[pd_train_ids, :]
            cont_spect = cont_spect[cont_train_ids, :, :, :]
            pd_spect = pd_spect[pd_train_ids, :, :, :]
        else:
            if fold is None:
                cont_test_ids = data['control']['test_unbalanced']
                pd_test_ids = data['pd']['test_unbalanced']
            else:
                cont_test_ids = data['control']['folds'][str(fold)]
                pd_test_ids = data['pd']['folds'][str(fold)]

            cont_meth = cont_meth.iloc[cont_test_ids, :]
            pd_meth = pd_meth.iloc[pd_test_ids, :]
            cont_meta = cont_meta.iloc[cont_test_ids, :]
            pd_meta = pd_meta.iloc[pd_test_ids, :]
            cont_spect = cont_spect[cont_test_ids, :]
            pd_spect = pd_spect[pd_test_ids, :]

        meth = pd.concat((cont_meth, pd_meth), axis=0).reset_index(drop=True).values
        metadata = pd.concat((cont_meta, pd_meta), axis=0).reset_index(drop=True)
        spect = np.concatenate((cont_spect, pd_spect), axis=0)
        age = metadata.Age # TODO need to account for this when augmenting

        metadata['diag_label'] = np.where(metadata['DIAGNOSIS'] == 'CONTROL', 0, 1)

        if balance and train:
            smt = SMOTE(k_neighbors=5)
            meth, class_ind = smt.fit_sample(meth, metadata.diag_label)
            augmenter = aug.OneOf(smt._row_ids.tolist(), image3d_transforms=[aug.image3d_augmenters.Rotate3d(angle=max_angle)])
            spect, _ = augmenter.apply_image3d(spect, metadata.diag_label)
        else:
            class_ind = metadata['diag_label']

        if augmentation and train and fraction > 0:
            augmenter = aug.OneOf(augment=fraction, image3d_transforms=[aug.image3d_augmenters.Rotate3d(angle=max_angle)],
                                  tabular_transforms=[aug.tabular_augmenters.GaussianPerturbation(fraction=0.05)])
            spect, class_ind = augmenter.apply_image3d(spect, class_ind)
            meth, _ = augmenter.apply_tabular(meth, class_ind)

        self.meth = torch.from_numpy(meth)
        self.spect = torch.from_numpy(spect)
        self.age = torch.from_numpy(np.array(age))
        self.class_ind = torch.from_numpy(np.array(class_ind))


    def __getitem__(self, idx):
        if self.classification:
            return self.meth[idx, :], self.spect[idx, :], self.class_ind[idx]
        else:
            return self.meth[idx, :], self.spect[idx, :], self.age[idx]

    def __len__(self):
        return len(self.meth)

class NoisyMNISTDataset(dataset.Dataset):
    def __init__(self, set="train"):
        self.set = set
        awgn = loadmat("data/mnist/mnist-with-awgn.mat")
        blur = loadmat("data/mnist/mnist-with-motion-blur.mat")
        contrast = loadmat("data/mnist/mnist-with-reduced-contrast-and-awgn.mat")

        self.awgn_x = awgn['{}_x'.format(set)]
        self.awgn_x = torch.from_numpy(np.array(np.expand_dims(self.awgn_x.reshape((self.awgn_x.shape[0], 28, 28)), axis=1)/255, dtype=np.float32))
        self.awgn_y = torch.from_numpy(awgn['{}_y'.format(set)].argmax(axis=1))
        self.blur_x = blur['{}_x'.format(set)]
        self.blur_x = torch.from_numpy(np.array(np.expand_dims(self.blur_x.reshape((self.blur_x.shape[0], 28, 28)), axis=1)/255, dtype=np.float32))
        self.blur_y = torch.from_numpy(blur['{}_y'.format(set)].argmax(axis=1))
        self.contrast_x = contrast['{}_x'.format(set)]
        self.contrast_x = torch.from_numpy(np.array(np.expand_dims(self.contrast_x.reshape((self.contrast_x.shape[0], 28, 28)), axis=1)/255, dtype=np.float32))
        self.contrast_y = torch.from_numpy(contrast['{}_y'.format(set)].argmax(axis=1))

    def __getitem__(self, idx):
        return self.awgn_x[idx, :], self.blur_x[idx, :], self.contrast_x[idx, :], self.awgn_y[idx]

    def __len__(self):
        return len(self.awgn_y)

class CIFAR10Dataset(dataset.Dataset):
    def __init__(self, train: bool, fmt: str = 'standard'):

        if train:
            setname = 'train'
        else:
            setname = 'test'
        with open("data/cifar10/{}_{}.pkl".format(setname, fmt), 'rb') as fd:
            data = pickle.load(fd)

        self.images = torch.from_numpy(np.array(data['X'].reshape((-1, 3, 32, 32))/255, dtype=np.float32))
        self.labels = torch.from_numpy(np.array(data['y']))

    def __getitem__(self, idx):
        return self.images[idx, :], self.labels[idx]

    def __len__(self):
        return len(self.images)

class MyMNISTDataset(dataset.Dataset):
    def __init__(self, train=True, awgn_frac=0.):
        self.set = set

        assert awgn_frac <= 1., "Noise cannot exceed 100%"

        if train:
            image_file = "data/mnist/train-images.idx3-ubyte"
            label_file = "data/mnist/train-labels.idx1-ubyte"
        else:
            image_file = "data/mnist/t10k-images.idx3-ubyte"
            label_file = "data/mnist/t10k-labels.idx1-ubyte"

        images = read_image_file(image_file)
        labels = read_label_file(label_file)

        seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=awgn_frac*255)])
        aug_images = seq.augment_images(images.numpy())

        self.mnist = torch.from_numpy(np.array(np.expand_dims(images, axis=1)/255, dtype=np.float32))
        self.aug_mnist = torch.from_numpy(np.array(np.expand_dims(aug_images, axis=1)/255, dtype=np.float32))
        self.labels = labels

    def __getitem__(self, idx):
        return self.mnist[idx, :], self.aug_mnist[idx, :], torch.zeros((2, 2)), self.labels[idx]

    def __len__(self):
        return len(self.labels)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

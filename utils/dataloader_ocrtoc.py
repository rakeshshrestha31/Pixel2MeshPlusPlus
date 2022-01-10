import numpy as np
import pickle
import json
import scipy.sparse as sp
# import networkx as nx
import threading
import queue
import sys
import cv2
import math
import time
import os
import glob


np.random.seed(123)


class DataFetcher(threading.Thread):
    def __init__(self, data_root, is_val=False, mesh_root=None, split='train'):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = queue.Queue(64)
        self.data_root = data_root
        self.is_val = is_val

        with open(os.path.join(data_root, 'train_test_split.json'), 'r') as f:
            split_data = json.load(f)[split]

        self.samples = [
            {
                'model_id': sample['model_id'],
                'images': images,
                'category': sample['category'],
                'triplet_id': triplet_id
            }
            for sample in split_data
            for triplet_id, images in enumerate(sample['images'])
        ]

        self.index = 0
        self.mesh_root = mesh_root
        self.number = len(self.samples)
        np.random.shuffle(self.samples)

    def work(self, idx):
        sample = self.samples[idx]
        pkl_item = '_'.join([
            sample['model_id'], str(sample['triplet_id']), 'gt_labels.dat'
        ])
        pkl_path = os.path.join(self.data_root, sample['model_id'], 'gt_labels.dat')
        pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
        if self.is_val:
            label = pkl[1]
        else:
            label = pkl
        # load image file
        category = sample['category']
        item_id = sample['model_id'] + '_' + str(sample['triplet_id'])

        img_path = os.path.join(
            self.data_root, sample['model_id'], 'segmented_color'
        )
        camera_meta_data = np.loadtxt(os.path.join(
            self.data_root, sample['model_id'], 'rendering_metadata.txt'
        ))

        if self.mesh_root is not None:
            mesh = np.loadtxt(os.path.join(self.mesh_root, category + '_' + item_id + '_00_predict.xyz'))
        else:
            mesh = None

        imgs = np.zeros((3, 224, 224, 3))
        poses = np.zeros((3, 16))
        for idx, view in enumerate(sample['images']):
            img = cv2.imread(
                os.path.join(img_path, f'segmented_color_{view}.png'),
                cv2.IMREAD_UNCHANGED
            )
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float32') / 255.0
            imgs[idx] = img_inp[:, :, :3]
            poses[idx] = camera_meta_data[int(view)]
        return imgs, label, poses, pkl_item, mesh

    def run(self):
        while self.index < 9000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.samples)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()


if __name__ == '__main__':
    data_root = sys.argv[1]
    data = DataFetcher(data_root)
    data.start()
    data.stopped = True

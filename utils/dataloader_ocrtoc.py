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
            for triplet_id, images in enumerate(sample['images'][:50])
            if sample['category'].lower() in ['sofa', 'chair'] # , 'cabinet/shelf/desk']
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
        T_world_cam0 = camera_meta_data[int(sample['images'][0])].reshape((4, 4))
        T_cam0_world = np.linalg.inv(T_world_cam0)

        for idx, view in enumerate(sample['images']):
            img = cv2.imread(
                os.path.join(img_path, f'segmented_color_{view}.png'),
                cv2.IMREAD_UNCHANGED
            )
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float32') / 255.0
            imgs[idx] = img_inp[:, :, :3]
            T_world_cam = camera_meta_data[int(view)].reshape((4, 4))
            T_cam0_cam = T_cam0_world @ T_world_cam
            poses[idx] = T_cam0_cam.reshape((-1))

        label_local = self.transform_label(label, poses[0])
        # self.debug(imgs, label, label_local, pkl_item)

        return imgs, label_local, poses, pkl_item, mesh

    @staticmethod
    def debug(imgs, label_world, label_local, data_id):
        X = label_local[:, 0]
        Y = label_local[:, 1]
        Z = label_local[:, 2]

        h = (193.3635 * np.divide(-Y, -Z)) + 112.0
        w = (193.3635 * np.divide(X, -Z)) + 112.0

        h = np.clip(h, 0, 223)
        w = np.clip(w, 0, 223)

        debug_img = imgs[0, ...]
        indices = (
            h.astype(np.uint16), w.astype(np.uint16),
            np.ones((h.shape[0], ), dtype=np.uint16) * 2
        )
        debug_img[indices] = 1.0
        debug_img = (debug_img * 255).astype(np.uint8)
        cv2.imwrite(f'{data_id}_debug_img.png', debug_img)

        with open(f'{data_id}_world.dat', 'wb') as f:
            pickle.dump(label_world, f)

        with open(f'{data_id}_local.dat', 'wb') as f:
            pickle.dump(label_local, f)

    @staticmethod
    def transform_label(label, T_world_cam):
        T_world_cam = T_world_cam.reshape((4, 4))
        T_cam_world = np.linalg.inv(T_world_cam)
        points_world = label[:, :3]
        normals_world = label[:, 3:]

        points_cam = T_cam_world @ DataFetcher.to_homogeneous(points_world).T
        points_cam = (points_cam.T)[:, :3]
        normals_cam = (T_cam_world[:3, :3] @ normals_world.T).T

        return np.concatenate(
            (points_cam, normals_cam), axis=-1
        )

    @staticmethod
    def to_homogeneous(x):
        return np.concatenate((x, np.ones_like(x[:, 0:1])), axis=-1)

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

# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import numpy as np
import pprint
import pickle
import shutil
import os

import open3d as o3d

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
# from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict, load_demo_image
# from utils.visualize import plot_scatter


def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-porcessing')
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # ---------------------------------------------------------------
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.placeholder(tf.float32, shape=(3, 224, 224, 3), name='img_inp'),
        'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],  # for laplace term
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],  # for unpooling
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
        'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
        'cameras': tf.placeholder(tf.float32, shape=(3, 16), name='Cameras'),
        'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }

    print('=> load data')
    # demo_img_list = ['data/demo/plane1.png',
    #                  'data/demo/plane2.png',
    #                  'data/demo/plane3.png']

    # demo_img_list = ['data/demo2/segmented_color_001.png',
    #                  'data/demo2/segmented_color_027.png',
    #                  'data/demo2/segmented_color_071.png']

    dataset_dir = '/home/rakesh/workspace/alibaba_3d_benchmark/3d-future/ocrtoc-rendered'

    # model_id = '065ce67a-19e8-4a6e-bb60-a23e64d214ee'
    # img_ids = [0, 24, 52]

    # model_id = 'a723b457-7c26-4cc8-916c-0355a5690fad'
    # img_ids = [1, 27, 71]

    # model_id = '70865842-b9d4-4b18-96b0-0cb8776b6f71'
    # img_ids = [3, 16, 28]

    model_id = '4c9c11ff-f385-4be3-bba4-ccb70ecba2df'
    img_ids = [28, 33, 58]

    demo_img_list = [os.path.join(
        dataset_dir, model_id, 'segmented_color',
        f'segmented_color_{img_id:03}.png'
    ) for img_id in img_ids]

    img_all_view = load_demo_image(demo_img_list)

    # cameras = np.loadtxt('data/demo/cameras.txt')
    cameras = np.loadtxt(
        os.path.join(
            dataset_dir, model_id,
            'rendering_metadata.txt'
        )
    )[img_ids]

    # data = DataFetcher(file_list=cfg.test_file_path, data_root=cfg.test_data_path, image_root=cfg.test_image_path, is_val=True)
    # data.setDaemon(True)
    # data.start()
    # ---------------------------------------------------------------

    # step = cfg.test_epoch
    # root_dir = os.path.join(cfg.save_path, cfg.name)
    model1_dir = os.path.join('results', 'coarse_mvp2m', 'models')
    model2_dir = os.path.join('results', 'refine_p2mpp', 'models')
    # predict_dir = os.path.join(cfg.save_path, cfg.name, 'predict', str(step))
    # if not os.path.exists(predict_dir):
    #     os.makedirs(predict_dir)
    #     print('==> make predict_dir {}'.format(predict_dir))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model1 = MVP2MNet(placeholders, logging=True, args=cfg)
    model2 = P2MPPNet(placeholders, logging=True, args=cfg)
    # ---------------------------------------------------------------
    print('=> initialize session')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    # sess2 = tf.Session(config=sesscfg)
    # sess2.run(tf.global_variables_initializer())
    # ---------------------------------------------------------------
    model1.load(sess=sess, ckpt_path=model1_dir, step=50)
    model2.load(sess=sess, ckpt_path=model2_dir, step=10)
    # exit(0)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    tflearn.is_training(False, sess)
    print('=> start test stage 1')
    feed_dict.update({placeholders['img_inp']: img_all_view})
    feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
    feed_dict.update({placeholders['cameras']: cameras})
    stage1_out3 = sess.run(model1.output3, feed_dict=feed_dict)

    print('=> start test stage 2')
    feed_dict.update({placeholders['features']: stage1_out3})
    vert = sess.run(model2.output2l, feed_dict=feed_dict)
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    face = np.loadtxt('data/face3.obj', dtype='|S32')
    mesh = np.vstack((vert, face))

    # pred_path = 'data/demo2/predict.obj'
    pred_path = '/tmp/predict.obj'
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

    print('=> save to {}'.format(pred_path))

    gt_path = os.path.join(dataset_dir, model_id, 'gt.obj')
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    T_world_cam0 = cameras[0].reshape((4, 4))
    gt_mesh = gt_mesh.transform(np.linalg.inv(T_world_cam0))
    o3d.io.write_triangle_mesh('/tmp/gt.obj', gt_mesh)


def main2():
    print('=> load data')

    dataset_dir = '/home/rakesh/workspace/alibaba_3d_benchmark/3d-future/ocrtoc-rendered'

    # model_id = '065ce67a-19e8-4a6e-bb60-a23e64d214ee'
    # img_ids = [0, 24, 52]

    # model_id = 'a723b457-7c26-4cc8-916c-0355a5690fad'
    # img_ids = [1, 27, 71]

    # model_id = '70865842-b9d4-4b18-96b0-0cb8776b6f71'
    # img_ids = [3, 16, 28]

    model_id = '4c9c11ff-f385-4be3-bba4-ccb70ecba2df'
    img_ids = [28, 33, 58]

    demo_img_list = [os.path.join(
        dataset_dir, model_id, 'segmented_color',
        f'segmented_color_{img_id:03}.png'
    ) for img_id in img_ids]

    img_all_view = load_demo_image(demo_img_list)

    # cameras = np.loadtxt('data/demo/cameras.txt')
    cameras = np.loadtxt(
        os.path.join(
            dataset_dir, model_id,
            'rendering_metadata.txt'
        )
    )[img_ids]

    gt_path = os.path.join(dataset_dir, model_id, 'gt.obj')
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=100000)

    placeholders = {
        'xyz': tf.placeholder(tf.float32, shape=(None, 3), name='xyz'),
        'camera_metadata': tf.placeholder(tf.float32, shape=(16, ), name='camera_metadata'),
    }

    print('=> build model')
    from utils.tools import camera_trans, camera_trans_inv
    def projection_model(xyz):
        X = xyz[:, 0]
        Y = xyz[:, 1]
        Z = xyz[:, 2]
        h = 193.3635 * tf.divide(-Y, -Z) + 112.0
        w = 193.3635 * tf.divide(X, -Z) + 112.0

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)

        return tf.stack([w, h], -1)

    projection_output = projection_model(placeholders['xyz'])
    points_local = camera_trans(**placeholders)
    points_world = camera_trans_inv(**placeholders)

    print('=> initialize session')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    tflearn.is_training(False, sess)

    feed_dict = {
        placeholders['xyz']: np.asarray(gt_pcd.points),
        placeholders['camera_metadata']: cameras[0]
    }
    points0 = sess.run(points_local, feed_dict=feed_dict)

    feed_dict.update({placeholders['xyz']: points0})
    points_origin = sess.run(points_world, feed_dict=feed_dict)


    projection_all_view = np.zeros_like(img_all_view)
    for cam_idx in range(cameras.shape[0]):
        feed_dict = {
            placeholders['xyz']: points_origin,
            placeholders['camera_metadata']: cameras[cam_idx]
        }

        points_current = sess.run(points_local, feed_dict=feed_dict)

        feed_dict = {
            placeholders['xyz']: points_current,
        }
        uv = sess.run(projection_output, feed_dict=feed_dict)
        uv = uv.astype(np.uint16)
        indices = (
            np.ones((uv.shape[0]), dtype=np.uint16) * cam_idx,
            uv[:, 1], uv[:, 0], # matrix indexing is [v, u]
            np.ones((uv.shape[0]), dtype=np.uint16) * 2, # red channel
        )
        projection_all_view[indices] = 1.0

    import cv2
    img_debug = []
    for img1, img2 in zip(img_all_view, projection_all_view):
        img_debug.append(cv2.addWeighted(img1, 1.0, img2, 0.7, 0))
    img_debug = np.concatenate(img_debug, axis=1)
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.imshow('win', img_debug)
    cv2.waitKey()


if __name__ == '__main__':
    main2()
    exit(0)

    print('=> set config')
    args=execute()
    # pprint.pprint(vars(args))
    main(args)


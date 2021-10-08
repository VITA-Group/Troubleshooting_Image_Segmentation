import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from collections import OrderedDict

import torch
from torchvision import transforms
import torchvision

from utils.utils import *
from utils.view_colors import color_map
from utils.metrics import compute_iou
from dataloaders.our_data_loader import dataloader
from dataloaders.voc import voc_dataloaders
from common_flags import COMMON_FLAGS


def find_object_sizes():
    '''
    Find the sizes of each object in PASCAL VOC 2012 testing set.

    Returns:
        object_size: disct. {1: [int], 2: [int], ..., 20: [int]}
    '''

    # check whether the json file already exists:
    save_json_file_path = os.path.join(COMMON_FLAGS.json_dir, 'object_sizes.json')
    if os.path.isfile(save_json_file_path):
        print('json file already exists: %s' % save_json_file_path)
        return

    # init dict:
    C = 21  # class number
    object_sizes = {}
    for object_id in range(1, C):  # 0 is background. 1-20 are useful objects. 255 is unlabeled.
        object_sizes[object_id] = []

    # dataloader:
    _, val_loader = voc_dataloaders(img_size=513, test_batch_size=10)

    for i, (imgs, segmaps) in enumerate(tqdm(val_loader)):
        segmaps = (segmaps * 255).int()
        # print('imgs:', imgs.size())
        # print('segmaps:', segmaps.size())
        for segmap in segmaps:
            object_ids = torch.unique(segmap).cpu().numpy()
            # print('object_ids:', object_ids)
            for object_id in object_ids:
                if object_id not in [0, 255]:  # we don't care about background class and unlabeled pixels
                    object_pixel_num = torch.sum(segmap == object_id).item()
                    # print('object_id:', object_id)
                    # print('object_pixel_num:', object_pixel_num)
                    object_sizes[object_id].append(object_pixel_num)

    # save json:
    with open(save_json_file_path, 'w+') as fp:
        json.dump(object_sizes, fp, sort_keys=True, indent='\t')

    return object_sizes


def find_imgs_with_object(model_name):
    '''
    Args:
        model: nn.Module

    Returns:
        imgs_with_object: dict. {0: [1, 2, 10], 1: [1, 3], ..., 20: [10, 11]}. Image 1.png contains both object 0 and 1.
    '''

    # check whether the json file already exists:
    save_json_file_path = os.path.join(COMMON_FLAGS.json_dir, 'imgs_with_object_%s.json' % model_name)
    if os.path.isfile(save_json_file_path):
        print('json file already exists: %s' % save_json_file_path)
        return

    # load object size:
    with open(os.path.join(COMMON_FLAGS.json_dir, 'object_sizes.json'), 'r') as fp:
        object_sizes = json.load(fp)
    print('object_sizes:', object_sizes.keys())

    # itialize thresholds:
    th_highs, th_lows = {}, {}
    for object_id in object_sizes:
        object_sizes_i = object_sizes[object_id]
        object_sizes_i = np.array(object_sizes_i)
        object_size_mean = np.mean(object_sizes_i)
        object_size_std = np.std(object_sizes_i)
        object_size_max = np.amax(object_sizes_i)
        object_size_min = np.amin(object_sizes_i)

        th_high = np.minimum(object_size_mean + 1 * object_size_std, object_size_max)
        th_low = np.maximum(object_size_mean - 1 * object_size_std, object_size_min)

        th_highs[object_id] = th_high
        th_lows[object_id] = th_low
    print('th_highs:', th_highs)
    print('th_lows:', th_lows)

    # initialize imgs_with_object dict:
    C = 21  # class number
    imgs_with_object = {}
    for object_id in range(1, C):  # 0 is background
        imgs_with_object[object_id] = []

    # get file names:
    file_name_list = os.listdir(COMMON_FLAGS.dataset_dir)
    N = len(file_name_list)

    # loop through each image:
    segmap_dir = os.path.join(COMMON_FLAGS.hdd_dir, 'pred_npy_%s' % model_name)
    for img_id in tqdm(range(N)):
        try:
            segmap = np.load(os.path.join(segmap_dir, 'result%d.npy' % img_id))
        except:
            print('something wrong with result%d.npy, skipped' % img_id)
            continue
        object_ids = np.unique(segmap)
        for object_id in object_ids:
            if object_id not in [0, 255]:  # we don't care about background class and unlabeled pixels
                object_pixel_num = np.sum(segmap == object_id)
                if th_lows[str(object_id)] <= object_pixel_num <= th_highs[str(object_id)]:
                    imgs_with_object[object_id].append(img_id)

    # save json:
    with open(save_json_file_path, 'w+') as fp:
        json.dump(imgs_with_object, fp, sort_keys=True, indent='\t')

    return imgs_with_object


def calculate_metrics(defender_model_name, attacker_model_name):
    '''
    Args:
        defender_model_name, attacker_model_name: strings.
    '''

    print('defender_model_name: %s\nattacker_model_name: %s' % (defender_model_name, attacker_model_name))

    # check whether the json file already exists:
    save_json_file_path = os.path.join(COMMON_FLAGS.json_dir,
                                       'mIoUs_%s_%s.json' % (defender_model_name, attacker_model_name))
    if os.path.isfile(save_json_file_path):
        print('json file already exists: %s' % save_json_file_path)
        return

    # load json:
    with open(os.path.join(COMMON_FLAGS.json_dir, 'imgs_with_object_%s.json' % defender_model_name), 'r') as fp:
        imgs_with_object_defender = json.load(fp)

    # init dict:
    C = 21  # class number
    mIoUs = {}
    for object_id in range(1, C):  # 0 is background. 1-20 are useful objects. 255 is unlabeled.
        mIoUs[object_id] = []

    # select images:
    C = 21  # class number
    for object_id in range(1, C):  # 0 is background. 1-20 are useful objects. 255 is unlabeled.
        print('Finding mIoU for images with object %d...' % object_id)
        imgs_with_object_i = imgs_with_object_defender[str(object_id)]
        # for img_id in tqdm(imgs_with_object_i):
        #     defender_segmap = np.load(os.path.join(defender_segmap_dir, 'result%d.npy' % img_id))
        #     attacker_segmap = np.load(os.path.join(attacker_segmap_dir, 'result%d.npy' % img_id))
        #     mIoU = compute_iou(defender_segmap.flatten(), attacker_segmap.flatten())
        #     mIoUs[object_id].append((img_id,mIoU))
        with Pool(48) as p:
            mIoUs[object_id] = list(tqdm(p.imap(_f, imgs_with_object_i), total=len(imgs_with_object_i)))

        # save json:
        with open(save_json_file_path, 'w+') as fp:
            json.dump(mIoUs, fp, sort_keys=True, indent='\t')

    return mIoUs


def MAD_selection(defender_model_name, attacker_model_name, top_n=10):
    '''
    Args:
        defender_model_name, attacker_model_name: strings.
    '''

    print('defender_model_name: %s\nattacker_model_name: %s' % (defender_model_name, attacker_model_name))

    # load json:
    with open(os.path.join(COMMON_FLAGS.json_dir,
                           'mIoUs_%s_%s.json' % (defender_model_name, attacker_model_name)), 'r') as fp:
        mIoUs = json.load(fp)

    # Find MAD images:
    top_mIoUs = OrderedDict()
    mIoUs = {int(k): v for k, v in mIoUs.items()}  # str keys to int keys
    for key in sorted(mIoUs.keys()):
        mIoU_i = mIoUs[key]
        mIoU_i = np.array(mIoU_i)
        mIoU_i = mIoU_i[np.argsort(mIoU_i[:,1])]
        top_mIoUs[key] = mIoU_i[0:top_n,:]
        print('mIoU_i:', mIoU_i.shape, mIoU_i[0:3,:])

    # save npy:
    np.save(os.path.join(COMMON_FLAGS.json_dir,
                         'top_mIoUs_%s_%s.npy' % (defender_model_name, attacker_model_name)), top_mIoUs)

if __name__ == '__main__':
    # find object sizes in PASCAL VOC val set:
    find_object_sizes()

    # Which images in our unlabeled dataset contains each object:
    model_name_list = ['deeplabv3_resnet101', 'fcn_resnet101']
    for model_name in model_name_list:
        find_imgs_with_object(model_name)

    # Find mIoU:
    other_model_name_list = ['fcn_resnet101']
    best_model_name = 'deeplabv3_resnet101'
    for other_model_name in other_model_name_list:
        # best model as defender or attacker:
        for defender_model_name, attacker_model_name in [(best_model_name, other_model_name),
                                                         (other_model_name, best_model_name)]:
            # def function for multiprocess:
            def _f(img_id):
                try:
                    defender_segmap = np.load(
                        os.path.join(COMMON_FLAGS.hdd_dir, 'pred_npy_%s' % defender_model_name, 'result%d.npy' % img_id)
                    )
                    attacker_segmap = np.load(
                        os.path.join(COMMON_FLAGS.hdd_dir, 'pred_npy_%s' % attacker_model_name, 'result%d.npy' % img_id)
                    )
                    mIoU = compute_iou(defender_segmap.flatten(), attacker_segmap.flatten())
                    return (img_id, mIoU)
                except:
                    print('something wrong with result%d.npy, skipped' % img_id)
                    return (img_id, -1)


            # select by MAD:
            calculate_metrics(defender_model_name, attacker_model_name)

            # select low mIoU images (maximum discrepancy):
            MAD_selection(defender_model_name, attacker_model_name, top_n=100)
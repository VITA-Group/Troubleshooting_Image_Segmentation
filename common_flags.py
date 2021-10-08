from argparse import Namespace
import os
from utils.utils import *

root_dir = os.path.join(os.path.expanduser('~'), 'MADA')
hdd_dir = '/hdd2/haotao/PASCALPlus'
json_dir = os.path.join(root_dir, 'jsons')
data_download_dir = os.path.join(hdd_dir, 'dataset')
dataset_dir = os.path.join(hdd_dir, 'dataset')
human_annotation_dir = os.path.join(hdd_dir, 'segmaps_bin')

create_dir(json_dir)

COMMON_FLAGS = Namespace()
COMMON_FLAGS.root_dir = root_dir
COMMON_FLAGS.hdd_dir = hdd_dir
COMMON_FLAGS.json_dir = json_dir
COMMON_FLAGS.data_download_dir = data_download_dir
COMMON_FLAGS.dataset_dir = dataset_dir
COMMON_FLAGS.human_annotation_dir = human_annotation_dir

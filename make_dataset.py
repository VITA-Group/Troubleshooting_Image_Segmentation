from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np 
import os
from shutil import copy2
from tqdm import tqdm
from utils.utils import create_dir

data_root_dir = os.path.join('/hdd2/haotao/PASCALPlus/dataset-2019-01-01-2020-01-01')
folder_list = [x[0] for x in os.walk(data_root_dir)][1:]
folder_list.sort()

merge_folder_name = os.path.join('/hdd2/haotao/PASCALPlus/dataset')
create_dir(merge_folder_name)

c = 0
for folder_idx, folder_name in enumerate(folder_list):
    print(folder_idx, folder_name)
    file_list = os.listdir(folder_name)
    for filename in tqdm(file_list):
        img = imread(os.path.join(folder_name, filename))
        if len(img.shape) == 3 and img.shape[-1]==3: # only consider RGB channel images.
            copy2(os.path.join(folder_name, filename), os.path.join(merge_folder_name, '%d.jpg' % c))
            c += 1
            # print(c)
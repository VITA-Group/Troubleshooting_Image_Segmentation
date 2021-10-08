'''
https://github.com/htwang14/MAD/blob/master/our_data_loader.py
'''  

import os
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from skimage.io import imread
from PIL import Image

from common_flags import COMMON_FLAGS

np.random.seed(5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_dir = os.path.join(COMMON_FLAGS.dataset_dir)
        # self.labels = np.loadtxt(os.path.join(COMMON_FLAGS.root_dir, 'dataset', 'labels.txt'))

    def __len__(self):
        # return len(self.labels)
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error:' + str(index)+'.jpg')
            item = self.load_item(0)

        return item
    
    def load_item(self, index):
        img = imread(os.path.join(self.img_dir, str(index)+'.jpg')) # [0,255] uint8.
        assert len(img.shape) == 3
        assert img.shape[-1] == 3
        # print('img:', type(img), img.shape)
        img = Image.fromarray(img)
        # print('img:', type(img), img.size)

        img = self.transform(img)
        # print('img:', type(img), img.size())

        # label = int(self.labels[index])

        # return img, label, index # img_file_name is str(index) + '.jpg'
        return img, index # img_file_name is str(index) + '.jpg'

def dataloader(batch_size, img_size=513, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_workers=8):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = Dataset(transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader
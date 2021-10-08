'''
Load hard samples (imgs and human seg labels) to finetune a model.
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

class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, transform, target_transform):
        super(FinetuneDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = os.path.join(COMMON_FLAGS.dataset_dir)
        self.labels_dir = os.path.join(COMMON_FLAGS.human_annotation_dir)
        self.label_filename_list = [f for f in os.listdir(self.labels_dir) if '.npy' in f]
        self.img_idx_list = []
        for label_filename in self.label_filename_list:
            img_idx = label_filename.split('segmap')[1]
            img_idx = int( img_idx.split('.npy')[0] )
            self.img_idx_list.append(img_idx)

    def __len__(self):
        return len(os.listdir(self.labels_dir))

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error:' + str(index)+'.jpg')
            item = self.load_item(0)

        return item
    
    def load_item(self, index):
        '''
        imgs: torch.Size([4, 3, 513, 513])
        labels: torch.Size([4, 1, 513, 513]) tensor([0.0078, 0.0235, 0.0275, 0.0549, 0.0588, 0.0706, 1.0000])
        idxes: tensor([5419, 8298, 8905,  777])
        '''
        # get img index:
        img_idx = self.img_idx_list[index]

        # read img:
        img = imread(os.path.join(self.img_dir, str(img_idx)+'.jpg')) # [0,255] uint8.
        assert len(img.shape) == 3
        assert img.shape[-1] == 3
        # print('img:', type(img), img.shape)
        img = Image.fromarray(img)
        # print('img:', type(img), img.size)
        img = self.transform(img)
        # print('img:', type(img), img.size())

        # load human annotation:
        label = np.load(os.path.join(self.labels_dir, 'segmap%d.npy' % img_idx))
        label = Image.fromarray(label)
        label = self.target_transform(label)

        return img, label # img_file_name is str(img_idx) + '.jpg'

def finetune_dataloader(batch_size, img_size=513, num_workers=8):
    test_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((img_size,img_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    test_dataset = FinetuneDataset(transform=test_transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader

if __name__ == '__main__':
    test_loader = finetune_dataloader(4)
    for imgs, labels in test_loader:
        print('imgs:', imgs.size())
        print('labels:', labels.size(), labels.unique())
        # print('idxes:', idxes)
        break

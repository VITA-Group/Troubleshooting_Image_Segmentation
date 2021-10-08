import argparse, os, time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
import torchvision

from utils.utils import *
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses
from utils.view_colors import color_map
from dataloaders.voc import voc_dataloaders

from models.deeplabv3plus.deeplab import DeepLab
from models.deeplabv3.deeplabv3 import resnet101

# args:
# training hyper params
parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--gpu', default='7')
parser.add_argument('--cpus', default=8, help='number of threads for data loader')
parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs to train')
parser.add_argument('--model_name', default='deeplabv3_resnet101', 
    choices=['deeplabv2_resnet101', 'deeplabv3_resnet101', 'deeplabv3plus_resnet101'])
parser.add_argument('--model_str', default='', help='which model to test')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True
gpu_num = torch.cuda.device_count()


# data:
_, val_loader = voc_dataloaders(test_batch_size=int(len(args.gpu)*8), num_workers=args.cpus)

# model:
if args.model_name == 'deeplabv2_resnet101':
    model_dir = 'models/deeplabv2_resnet101/'
    model_path = os.path.join(model_dir, "deeplabv2_resnet101_msc-vocaug-20000.pth")
    model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", n_classes=21) # OS=8
    model.load_state_dict(torch.load(model_path))
elif args.model_name == 'deeplabv3plus_resnet101':
    model_dir = 'models/deeplabv3plus/'
    model_path = os.path.join(model_dir, "deeplab-resnet.pth")
    model = DeepLab(num_classes=21, backbone='resnet', sync_bn=False, freeze_bn=False) # OS=16
    model.load_state_dict(torch.load(model_path)['state_dict'])
elif args.model_name == 'deeplabv3_resnet101':
    model = resnet101(num_classes=21)
model = nn.DataParallel(model).cuda()
# load pth:
if args.model_str:
    model_pth_path = os.path.join(args.model_str, 'best_mIoU.pth')
    model.load_state_dict(torch.load(model_pth_path)['model'])
    print('ckpt loaded from %s...' % model_pth_path)
else:
    print('using original ckpt...')

# loss function:
criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')

# evaluator:
evaluator = Evaluator(21)

# eval:
with torch.no_grad():
    model.eval()
    val_loss = AverageMeter()
    evaluator.reset()
    for i, (imgs, targets) in enumerate(tqdm(val_loader)):
        # data on cuda:
        imgs = imgs.cuda()
        targets = ( targets * 255 ).int() # [0,1] -> [0,255]
        targets = targets.cuda().squeeze(1)

        # forward:
        if args.model_name in ['deeplabv2_resnet101', 'deeplabv3_resnet101', 'deeplabv3plus_resnet101']:
            logits = model(imgs)
        # reshape logits as the same size as targets:
        logits = F.interpolate(logits, (513, 513), mode='bilinear')
        loss = criterion(logits, targets)

        # metrics:
        _, preds = torch.max(logits, dim=1)
        evaluator.add_batch(targets.cpu().numpy(), preds.data.cpu().numpy())

        # append:
        val_loss.append(loss.item())

pixel_acc = evaluator.Pixel_Accuracy()
class_acc = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
IoUs = evaluator.Intersection_over_Union()

# print:
result_str = 'loss %.4f | mIoU %.4f | pixel_acc %.4f | class_acc %.4f \nIoUs %s' % (
    val_loss.avg, mIoU, pixel_acc, class_acc, IoUs)
print(result_str)

# save txt files:
with open(os.path.join(model_dir, 'results.txt'), 'a+') as fp:
    fp.write(result_str)

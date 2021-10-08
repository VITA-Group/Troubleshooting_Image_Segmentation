import argparse, os, time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn 
from torchvision import transforms
import torchvision

from utils.utils import *
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses
from utils.view_colors import color_map
from dataloaders.voc import voc_dataloaders
from dataloaders.finetune_dataloader import finetune_dataloader

# args:
# training hyper params
parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--gpu', default='7')
parser.add_argument('--cpus', default=8, help='number of threads for data loader')
parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=None, help='input batch size for training')
# optimizer params
parser.add_argument('--lr', type=float, default=0.007, help='learning rate')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='w-decay (default: 5e-4)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True
if args.batch_size is None:
    gpu_num = torch.cuda.device_count()
    args.batch_size = 2 * gpu_num

# color map of VOC:
cmap = color_map()

# mkdir:
opt_str = 'e%d-b%d-sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
model_str = '%s' % (opt_str)
save_dir = os.path.join('results_finetune', model_str)
create_dir(save_dir)

# data:
train_loader, val_loader = voc_dataloaders(
    train_batch_size=args.batch_size, test_batch_size=int(len(args.gpu)*8), num_workers=args.cpus)
# infinite repeater:
from itertools import repeat
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data, labels in loader:
            yield data, labels
train_iter = repeater(train_loader)
finetune_loader = finetune_dataloader(batch_size=args.batch_size, num_workers=args.cpus)

# model:
# model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
backbone_parameters, clf_parameters, aux_parameters = [], [], []
for pname, p in model.named_parameters():
    print(pname, p.size())
    if 'backbone' in pname:
        backbone_parameters.append(p)
    elif 'classifier' in pname:
        clf_parameters.append(p)
    elif 'aux' in pname:
        aux_parameters.append(p)
model = nn.DataParallel(model).cuda()

# opt:
optimizer = torch.optim.SGD(
    [{'params': backbone_parameters, 'lr': args.lr},
    {'params': clf_parameters + aux_parameters, 'lr': args.lr * 10}], 
    momentum=args.momentum, weight_decay=args.wd, nesterov=True)

# scheduler:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
# TODO

# loss function:
criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')

# evaluator:
evaluator = Evaluator(21)

# training loop:
train_losses, val_losses, val_mIoUs, val_pixel_accs, val_class_accs = [], [], [], [], []
best_mIoU = 0
max_iter = args.epochs * len(train_loader)
for epoch in range(args.epochs):
    start_time = time.time()

    ## train:
    model.train()
    train_loss = AverageMeter()
    for i, (inputs, targets) in enumerate(finetune_loader):
        # lr scheduler:
        current_iter = epoch * len(train_loader) + i
        current_lr = args.lr * (1 - float(current_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = current_lr
        optimizer.param_groups[1]['lr'] = current_lr * 10

        # get a batch from train loader:
        inputs_voc, targets_voc = next(train_iter)
        # concate:
        inputs = torch.cat([inputs, inputs_voc], dim=0)
        targets = torch.cat([targets, targets_voc], dim=0)

        # data on cuda:
        inputs = inputs.cuda()
        targets = ( targets * 255 ).int() # [0,1] -> [0,255]
        targets = targets.cuda().squeeze(1)

        # forward:
        outputs = model(inputs)['out']
        loss = criterion(outputs, targets)
        
        # backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # append:
        train_loss.append(loss.item())

        # print:
        if i % 10 == 0:
            print('epoch %d-batch %d/%d (train): loss %.4f | current_lr %.4f' % (
                epoch, i, len(train_loader), loss.item(), current_lr))

        # break
        
    
    ## eval:
    if epoch % 5 == 0 or epoch >= int(0.75*args.epochs):
        with torch.no_grad():
            model.eval()
            val_loss = AverageMeter()
            evaluator.reset()
            for i, (inputs, targets) in enumerate(tqdm(val_loader)):
                # data on cuda:
                inputs = inputs.cuda()
                targets = ( targets * 255 ).int() # [0,1] -> [0,255]
                targets = targets.cuda().squeeze(1)

                # forward:
                outputs = model(inputs)['out']
                loss = criterion(outputs, targets)

                # metrics:
                preds = np.argmax(outputs.data.cpu().numpy(), axis=1)
                evaluator.add_batch(targets.cpu().numpy(), preds)

                # append:
                val_loss.append(loss.item())

        pixel_acc = evaluator.Pixel_Accuracy()
        class_acc = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()

        # print:
        print('epoch %d/%d (time %.2f): loss %.4f | mIoU %.4f (bset %.4f) | pixel_acc %.4f | class_acc %.4f' % (
            epoch, args.epochs, time.time()-start_time, val_loss.avg, 
            mIoU, best_mIoU, pixel_acc, class_acc))


    # save loss curve:
    train_losses.append(train_loss.avg)
    plt.plot(train_losses, 'r')
    val_losses.append(val_loss.avg)
    plt.plot(val_losses, 'g')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()

    val_pixel_accs.append(pixel_acc)
    plt.plot(val_pixel_accs, 'g')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_pixel_accs.png'))
    plt.close()

    val_class_accs.append(class_acc)
    plt.plot(val_class_accs, 'g')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_class_accs.png'))
    plt.close()

    val_mIoUs.append(mIoU)
    plt.plot(val_mIoUs, 'g')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_mIoUs.png'))
    plt.close()

    # save models:
    model.eval()
    if mIoU > best_mIoU:
        best_mIoU = mIoU
        torch.save(
            {'epoch': epoch, 'best_mIoU': best_mIoU, 
            'model': model.state_dict(), 'opt': optimizer.state_dict()},
            os.path.join(save_dir, 'best_mIoU.pth')
        )

    torch.save(
        {'epoch': epoch, 'best_mIoU': best_mIoU, 
        'model': model.state_dict(), 'opt': optimizer.state_dict()},
        os.path.join(save_dir, 'latest.pth')
    )
    


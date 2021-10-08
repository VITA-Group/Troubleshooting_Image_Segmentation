from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Subset
from PIL import Image

def voc_dataloaders(img_size=513, train_batch_size=4, test_batch_size=4, num_workers=4, data_dir = 'datasets/voc'):

    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(), # first ToTensor, then Normalize!
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((img_size,img_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    train_set = VOCSegmentation(data_dir, image_set='train',
                                transform=target_transform, target_transform=target_transform, download=True)
    val_set = VOCSegmentation(data_dir, image_set='val',
                              transform=target_transform, target_transform=target_transform, download=False)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


if __name__ == '__main__':
    from tqdm import tqdm
    train_loader, val_loader = voc_dataloaders(img_size=513, train_batch_size=1, test_batch_size=1)

    # for i, (imgs, segmaps) in enumerate(tqdm(train_loader)):
    #     pass

    for i, (imgs, segmaps) in enumerate(tqdm(val_loader)):
        pass
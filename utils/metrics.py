from sklearn.metrics import confusion_matrix  
import numpy as np
import os 
from tqdm import tqdm 

def compute_iou(y_pred, y_true):
    ''' from 
    https://stackoverflow.com/questions/31653576/how-to-calculate-the-mean-iu-score-in-image-segmentation
    '''
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=np.arange(21))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection[union!=0] / union[union!=0].astype(np.float32)
    mIoU = np.mean(IoU)
    return mIoU

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoUs = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoUs

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    # find IoU for all images:
    model_name1 = 'fcn_resnet101'
    model_name2 = 'deeplabv3_resnet101'

    data_root_dir = os.path.join('/hdd2/haotao/PASCALPlus/dataset')
    pred_npy_root_dir1 = os.path.join('/hdd2/haotao/PASCALPlus/pred_npy_%s' % model_name1)
    pred_npy_root_dir2 = os.path.join('/hdd2/haotao/PASCALPlus/pred_npy_%s' % model_name2)

    mIoU_list = []
    file_list = len(os.listdir(data_root_dir))
    for i in tqdm(range(file_list)):

        filename = '%d.jpg' % i
        filepath = os.path.join(data_root_dir, filename)
        if not os.path.isfile(filepath): # maybe some images are not properly downloaded.
            print('file not exist: %s' % filepath)
            continue
    
        output_prediction1 = np.load(os.path.join(pred_npy_root_dir1, 'result%d.npy' % i))
        output_prediction2 = np.load(os.path.join(pred_npy_root_dir2, 'result%d.npy' % i))

        output_prediction1 = output_prediction1.flatten()
        output_prediction2 = output_prediction2.flatten()

        mIoU = compute_iou(output_prediction1, output_prediction2)
        mIoU_list.append(mIoU)

        if i % 100 == 0:
            np.save('mIoU_list_%s_%s.npy' % (model_name1, model_name2), np.array(mIoU_list))

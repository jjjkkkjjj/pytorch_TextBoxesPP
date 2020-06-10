from text_data import datasets

from ssd.models.ssd300 import SSD300
from ssd.train import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import torch

if __name__ == '__main__':
    """
    augmentation = augmentations.AugmentationOriginal()
    #augmentation = None

    transform = transforms.Compose(
        [transforms.Resize((300, 300)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.ToCentroids(),
         target_transforms.OneHot(class_nums=datasets.VOC_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )
    """
    train_dataset = datasets.COCO2014Text_Dataset(ignore=None, transform=None, target_transform=None, augmentation=None)
    train_dataset[1]
    train_dataset[100]
    train_dataset[150]
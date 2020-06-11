from text_data import datasets, target_transforms, transforms

from ssd.models.ssd300 import SSD300
from textboxespp.models.textboxespp import TextBoxesPP
from ssd.train import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

from torch.optim.sgd import SGD
import torch

if __name__ == '__main__':

    #augmentation = augmentations.AugmentationOriginal()
    #augmentation = None

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Corners2Centroids(),
         target_transforms.ToQuadrilateral(),
         target_transforms.OneHot(class_nums=datasets.COCOText_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )

    train_dataset = datasets.COCO2014Text_Dataset(ignore=target_transforms.Ignore(illegible=True), transform=transform, target_transform=target_transform, augmentation=None)
    train_dataset[1]
    """
    k = TextBoxesPP()
    aa = k.state_dict()
    a = torch.load('./weights/model_icdar15.pth')
    k=0
    """
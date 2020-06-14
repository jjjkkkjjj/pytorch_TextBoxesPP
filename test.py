from text_data import datasets, target_transforms, transforms, augmentations
from text_data.utils import batch_ind_fn_droptexts

from textboxespp.models.textboxespp import TextBoxesPP

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
import cv2
import torch

if __name__ == '__main__':

    augmentation = augmentations.RandomSampled()
    #augmentation = None

    transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Corners2Centroids(),
         #target_transforms.ToQuadrilateral(),
         target_transforms.OneHot(class_nums=datasets.COCOText_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )

    #train_dataset = datasets.COCO2014Text_Dataset(ignore=target_transforms.Ignore(illegible=True), transform=transform, target_transform=target_transform, augmentation=None)
    test_dataset = datasets.SynthTextDataset(ignore=None, transform=transform, target_transform=target_transform, augmentation=augmentation)

    model = TextBoxesPP(input_shape=(384, 384, 3)).cuda()
    model.load_vgg_weights()
    print(model)
    #model.load_weights('./weights/model_icdar15.pth')
    model.load_weights('weights/checkpoints/pretrained-synthtext_i-0050000_checkpoints20200614.pth')
    model.eval()

    images = [test_dataset[i][0] for i in range(20)]
    inf, ret_imgs = model.infer(images, visualize=True, toNorm=False)
    for img in ret_imgs:
        cv2.imshow('result', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()
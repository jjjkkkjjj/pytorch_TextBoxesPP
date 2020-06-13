from text_data import datasets, target_transforms, transforms, augmentations
from text_data.utils import batch_ind_fn_droptexts

from textboxespp.models.textboxespp import TextBoxesPP
from ssd.train import *
from textboxespp.train.loss import TextBoxLoss

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

from torch.optim.sgd import SGD
import torch

if __name__ == '__main__':

    augmentation = augmentations.RandomSampled()
    #augmentation = None

    transform = transforms.Compose(
        [transforms.Resize((768, 768)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Corners2Centroids(),
         target_transforms.ToQuadrilateral(),
         target_transforms.OneHot(class_nums=datasets.COCOText_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )

    #train_dataset = datasets.COCO2014Text_Dataset(ignore=target_transforms.Ignore(illegible=True), transform=transform, target_transform=target_transform, augmentation=None)
    train_dataset = datasets.SynthTextDataset(ignore=None, transform=transform, target_transform=target_transform, augmentation=augmentation)

    k = TextBoxesPP()
    aa = k.state_dict()
    a = torch.load('./weights/model_icdar15.pth')
    k=0


    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              collate_fn=batch_ind_fn_droptexts,
                              num_workers=4,
                              pin_memory=True)

    model = TextBoxesPP().cuda()
    print(model)
    #model.load_vgg_weights()
    #model.load_weights('./weights/model_icdar15.pth')

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    # iter_sheduler = SSDIterMultiStepLR(optimizer, milestones=(10, 20, 30), gamma=0.1, verbose=True)
    iter_sheduler = SSDIterStepLR(optimizer, step_size=60000, gamma=0.1, verbose=True)

    save_manager = SaveManager(modelname='ssd300', interval=5000, max_checkpoints=3)
    log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=None)
    trainer = TrainLogger(model, loss_func=TextBoxLoss(alpha=0.2), optimizer=optimizer, scheduler=iter_sheduler,
                          log_manager=log_manager)

    trainer.train(80000, train_loader)  # , evaluator=VOC2007Evaluator(val_dataset, iteration_interval=10))


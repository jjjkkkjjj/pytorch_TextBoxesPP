from ssd.models.base import *

class TextBoxesPP(SSDvggBase):
    def __init__(self, input_shape=(1024, 1024, 3),
                 val_config=SSDValConfig(val_conf_threshold=0.01, vis_conf_threshold=0.6, iou_threshold=0.45, topk=200)):
        """
        :param input_shape:
        :param val_config:
        """
        train_config = SSDTrainConfig(class_labels=('text',), input_shape=input_shape, batch_norm=False,

                                      aspect_ratios=((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
                                      classifier_source_names=(
                                      'convRL4_3', 'convRL7', 'convRL8_2', 'convRL9_2', 'convRL10_2', 'convRL11_2'),
                                      addon_source_names=('convRL4_3',),

                                      codec_means=(0.0, 0.0, 0.0, 0.0), codec_stds=(0.1, 0.1, 0.2, 0.2),
                                      rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))

        ### layers ###
        Conv2d.batch_norm = False
        vgg_layers = [
            *Conv2d.relu_block('1', 2, train_config.input_channel, 64),

            *Conv2d.relu_block('2', 2, 64, 128),

            *Conv2d.relu_block('3', 3, 128, 256, pool_ceil_mode=True),

            *Conv2d.relu_block('4', 3, 256, 512),

            *Conv2d.relu_block('5', 3, 512, 512, pool_k_size=(3, 3), pool_stride=(1, 1), pool_padding=1),
            # replace last maxpool layer's kernel and stride

            # Atrous convolution
            *Conv2d.relu_one('6', 512, 1024, kernel_size=(3, 3), padding=6, dilation=6),

            *Conv2d.relu_one('7', 1024, 1024, kernel_size=(1, 1)),
        ]

        extra_layers = [
            *Conv2d.relu_one('8_1', 1024, 256, kernel_size=(1, 1)),
            *Conv2d.relu_one('8_2', 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),

            *Conv2d.relu_one('9_1', 512, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('9_2', 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),

            *Conv2d.relu_one('10_1', 256, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('10_2', 128, 256, kernel_size=(3, 3)),

            *Conv2d.relu_one('11_1', 256, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('11_2', 128, 256, kernel_size=(3, 3), batch_norm=False),
            # if batch_norm = True, error is thrown. last layer's channel == 1 may be caused
        ]
        vgg_layers = nn.ModuleDict(vgg_layers)
        extra_layers = nn.ModuleDict(extra_layers)

        super().__init__(train_config, val_config, defaultBox=DBoxSSDOriginal(img_shape=input_shape,
                                                                              scale_conv4_3=0.1, scale_range=(0.2, 0.9),
                                                                              aspect_ratios=train_config.aspect_ratios),
                         vgg_layers=vgg_layers, extra_layers=extra_layers)

    def build_classifier(self, **kwargs):
        """
        override build_classifier because kernel size is different from original one
        :param kwargs:
        :return:
        """
        # loc and conf layers
        in_channels = tuple(self.feature_layers[name].out_channels for name in self.classifier_source_names)

        _dbox_num_per_fpixel = [len(aspect_ratio) * 2 for aspect_ratio in self.aspect_ratios]
        # loc
        # dbox_num * 2=(original and "with vertical offset") * 12(=cx,cy,w,h,x1,y1,x2,y2,...)
        # note that the reason of multiplying 2 of dbox_num *2 is for default boxes with vertical offset
        out_channels = tuple(dbox_num * 2 * 12 for dbox_num in _dbox_num_per_fpixel)
        localization_layers = [
            *Conv2d.block('_loc', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 5),
                          padding=(1, 2), batch_norm=False)
        ]
        self.localization_layers = nn.ModuleDict(OrderedDict(localization_layers))

        # conf
        # dbox_num * 2=(original and "with vertical offset") * 2(=text or background)
        # note that the reason of multiplying 2 of dbox_num *2 is for default boxes with vertical offset
        out_channels = tuple(dbox_num * 2 * 2 for dbox_num in _dbox_num_per_fpixel)
        confidence_layers = [
            *Conv2d.block('_conf', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 5),
                          padding=(1, 2), batch_norm=False)
        ]
        self.confidence_layers = nn.ModuleDict(OrderedDict(confidence_layers))

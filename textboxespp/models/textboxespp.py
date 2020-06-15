from ssd.models.base import *
from ssd._utils import _check_ins
from ..core.boxes.dbox import DBoxTextBoxOriginal
from ..core.boxes.codec import TextBoxCodec
from ..core.predictor import TextBoxPredictor
from ..core.inference import InferenceBox, textbox_non_maximum_suppression
from ..core.inference import toVisualizeInfQuadsRGBimg

class TextBoxesPPValConfig(SSDValConfig):
    def __init__(self, **kwargs):
        self.iou_threshold2 = _check_ins('iou_threshold2', kwargs.get('iou_threshold2', 0.2), float)
        super().__init__(**kwargs)

class TextBoxesPP(SSDvggBase):
    def __init__(self, input_shape=(768, 768, 3),
                 val_config=TextBoxesPPValConfig(val_conf_threshold=0.01, vis_conf_threshold=0.6,
                                                 iou_threshold=0.5, iou_threshold2=0.2, topk=200)):
        """
        :param input_shape:
        :param val_config:
        """
        train_config = SSDTrainConfig(class_labels=('text',), input_shape=input_shape, batch_norm=False,

                                      aspect_ratios=((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
                                      classifier_source_names=(
                                      'convRL4_3', 'convRL7', 'convRL8_2', 'convRL9_2', 'convRL10_2', 'convRL11_2'),
                                      addon_source_names=('convRL4_3',),

                                      codec_means=(0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0),
                                      codec_stds=(0.1, 0.1, 0.2, 0.2,
                                                  0.1, 0.1, 0.1, 0.1,
                                                  0.1, 0.1, 0.1, 0.1),
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

        super().__init__(train_config, val_config, defaultBox=DBoxTextBoxOriginal(img_shape=input_shape,
                                                                                  scale_conv4_3=0.1, scale_range=(0.2, 0.9),
                                                                                  aspect_ratios=train_config.aspect_ratios),

                         codec=TextBoxCodec(norm_means=train_config.codec_means, norm_stds=train_config.codec_stds),
                         predictor=TextBoxPredictor(2),
                         inferenceBox=InferenceBox(2, filter_func=textbox_non_maximum_suppression, val_config=val_config),

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

    def infer(self, image, conf_threshold=None, toNorm=False, visualize=False):
        """
        :param image: ndarray or Tensor of list or tuple, or ndarray, or Tensor. Note that each type will be handled as;
            ndarray of list or tuple, ndarray: (?, h, w, c). channel order will be handled as RGB
            Tensor of list or tuple, Tensor: (?, c, h, w). channel order will be handled as RGB
        :param conf_threshold: float or None, if it's None, default value will be passed
        :param toNorm: bool, whether to normalize passed image
        :param visualize: bool,
        :return:
        """
        if not self.isBuilt:
            raise NotImplementedError(
                "Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")
        if self.training:
            raise NotImplementedError("call \'eval()\' first")

        # img: Tensor, shape = (b, c, h, w)
        img = check_image(image, self.device)

        # normed_img, orig_img: Tensor, shape = (b, c, h, w)
        normed_img, orig_img = get_normed_and_origin_img(img, self.rgb_means, self.rgb_stds, toNorm, self.device)

        if list(img.shape[1:]) != [self.input_channel, self.input_height, self.input_width]:
            raise ValueError('image shape was not same as input shape: {}, but got {}'.format(
                [self.input_channel, self.input_height, self.input_width], list(img.shape[1:])))

        if conf_threshold is None:
            conf_threshold = self.vis_conf_threshold if visualize else self.val_conf_threshold

        with torch.no_grad():

            # predict
            predicts = self(normed_img)

            predicts = self.decoder(predicts, self.dboxes)

            # list of tensor, shape = (box num, 14=(class index, confidence, cx, cy, w, h, 12=(x1, y1,...)))
            infers = self.inferenceBox(predicts, conf_threshold)

            img_num = normed_img.shape[0]
            if visualize:
                return infers, [toVisualizeInfQuadsRGBimg(orig_img[i], poly_pts=infers[i][:, 6:], inf_labels=infers[i][:, 0],
                                                          inf_confs=infers[i][:, 1], classe_labels=self.class_labels,
                                                          verbose=False) for i in range(img_num)]
            else:
                return infers


    def load_vgg_weights(self):
        if self.batch_norm:
            load_vgg_weights(self, 'vgg16_bn')
        else:
            load_vgg_weights(self, 'vgg16')
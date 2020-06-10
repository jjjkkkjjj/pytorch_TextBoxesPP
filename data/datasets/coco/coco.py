import cv2, os
import numpy as np
from .coco_text import COCO_Text
from ssd_data.datasets import coco

from ssd_data._utils import DATA_ROOT, _check_ins

COCO_class_labels = []
len(COCO_class_labels)

class COCOTextSingleDatasetBase(coco.ObjectDetectionDatasetBase):
    def __init__(self, coco_dir, focus, ignore=None, transform=None, target_transform=None, augmentation=None,
                 class_labels=None):
        """
        :param coco_dir: str, coco directory path above 'annotations' and 'images'
                e.g.) coco_dir = '~~~~/coco2007/trainval'
        :param focus: str or str, directory name under images
                e.g.) focus = 'train2014'
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        :param class_labels: None or list or tuple, if it's None use VOC_class_labels
        """
        super().__init__(ignore=ignore, transform=transform, target_transform=target_transform,
                         augmentation=augmentation)

        self._coco_dir = coco_dir
        self._focus = focus

        self._class_labels = _check_ins('class_labels', class_labels, (list, tuple), allow_none=True)
        if self._class_labels is None:
            self._class_labels = COCO_class_labels

        self._annopath = os.path.join(self._coco_dir, 'annotations', 'instances_' + self._focus + '.json')
        if os.path.exists(self._annopath):
            self._coco = COCO_Text(self._annopath)
        else:
            raise FileNotFoundError('json: {} was not found'.format('instances_' + self._focus + '.json'))

        # remove no annotation image
        self._imageids = list(self._coco.imgToAnns.keys())
        self._get_image(0)
        self._get_bbox_lind(0)

    @property
    def class_nums(self):
        return len(self._class_labels)

    @property
    def class_labels(self):
        return self._class_labels

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', self._focus, filename)

    def __len__(self):
        return len(self._imageids)

    """
    Detail of contents in voc > https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    VOC bounding box (xmin, ymin, xmax, ymax)
    """

    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(ndarray)
        """

        """
        self._coco.loadImgs(self._imageids[index]): list of dict, contains;
            license: int
            file_name: str
            coco_url: str
            height: int
            width: int
            date_captured: str
            flickr_url: str
            id: int
        """
        filename = self._coco.loadImgs(self._imageids[index])[0]['file_name']
        img = cv2.imread(self._jpgpath(filename))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    def _get_bbox_lind(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated,...])
        """
        linds = []
        bboxes = []
        flags = []

        # anno_ids is list
        anno_ids = self._coco.getAnnIds(self._imageids[index])

        # annos is list of dict
        annos = self._coco.loadAnns(anno_ids)
        for anno in annos:
            """
            anno's  keys are;
                segmentation: list of float
                area: float
                iscrowd: int, 0 or 1
                image_id: int
                bbox: list of float, whose length is 4
                category_id: int
                id: int
            """
            """
            self._coco.loadCats(anno['category_id']) is list of dict, contains;
                supercategory: str
                id: int
                name: str
            """
            cat = self._coco.loadCats(anno['category_id'])[0]

            linds.append(self.class_labels.index(cat['name']))

            # bbox = [xmin, ymin, w, h]
            xmin, ymin, w, h = anno['bbox']
            # convert to corners
            xmax, ymax = xmin + w, ymin + h
            bboxes.append([xmin, ymin, xmax, ymax])

            """
            flag = {}
            keys = ['iscrowd']
            for key in keys:
                if key in anno.keys():
                    flag[key] = anno[key] == 1
                else:
                    flag[key] = False
            flags.append(flag)
            """
            flags.append({'difficult': anno['iscrowd'] == 1})

        return np.array(bboxes, dtype=np.float32), np.array(linds, dtype=np.float32), flags


class COCO
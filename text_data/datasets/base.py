from ssd_data.datasets.base import ObjectDetectionDatasetBase as ObjectDetectionDatasetBase
import torch
import numpy as np

class TextDetectionDatasetBase(ObjectDetectionDatasetBase):
    def __getitem__(self, index):
        """
        :param index: int
        :return:
            img : rgb image(Tensor or ndarray)
            targets and texts:
                targets : Tensor or ndarray of bboxes and labels [box, label]
                = [xmin, ymin, xmamx, ymax, label index(or relu_one-hotted label)]
                or
                = [cx, cy, w, h, label index(or relu_one-hotted label)]
                texts: list of str, if it's illegal, str = ''
        """
        img = self._get_image(index)
        bboxes, linds, flags, texts = self._get_target(index)

        img, bboxes, linds, flags, texts = self.apply_transform(img, bboxes, linds, flags, texts)

        # concatenate bboxes and linds
        if isinstance(bboxes, torch.Tensor) and isinstance(linds, torch.Tensor):
            if linds.ndim == 1:
                linds = linds.unsqueeze(1)
            targets = torch.cat((bboxes, linds), dim=1)
        else:
            if linds.ndim == 1:
                linds = linds[:, np.newaxis]
            targets = np.concatenate((bboxes, linds), axis=1)

        return img, (targets, texts)
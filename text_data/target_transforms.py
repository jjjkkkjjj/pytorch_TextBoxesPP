import logging
from ssd_data.target_transforms import Ignore as _Ignore
from ssd_data._utils import _check_ins

from ssd_data.target_transforms import *

class Ignore(_Ignore):
    supported_key = ['illegible', 'difficult']

    def __init__(self, **kwargs):
        """
        :param kwargs: if true, specific keyword will be ignored
        """
        self.ignore_key = []
        for key, val in kwargs.items():
            if key in Ignore.supported_key:
                val = _check_ins(key, val, bool)
                if not val:
                    logging.warning('No meaning: {}=False'.format(key))
                else:
                    self.ignore_key += [key]
            else:
                logging.warning('Unsupported arguments: {}'.format(key))

    def __call__(self, bboxes, labels, flags, *args):
        ret_bboxes = []
        ret_labels = []
        ret_flags = []
        ret_quads = []
        ret_texts = []

        quads = args[0]
        texts = args[1]

        for bbox, label, flag, quad, text in zip(bboxes, labels, flags, quads, texts):
            flag_keys = list(flag.keys())
            ig_flag = [flag[ig_key] if ig_key in flag_keys else False for ig_key in self.ignore_key]
            if any(ig_flag):
                continue
            """
            isIgnore = False
            for key, value in self.kwargs.items():
                if value and key in flag and flag[key]:
                    isIgnore = True
                    break
            if isIgnore:
                continue
            #if self._ignore_partial and flag['partial']:
            #    continue
            """
            # normalize
            # bbox = [xmin, ymin, xmax, ymax]
            ret_bboxes += [bbox]
            ret_labels += [label]
            ret_flags += [flag]
            ret_quads += [quad]
            ret_texts += [text]

        ret_bboxes = np.array(ret_bboxes, dtype=np.float32)
        ret_labels = np.array(ret_labels, dtype=np.float32)
        ret_quads = np.array(ret_quads, dtype=np.float32)

        return ret_bboxes, ret_labels, ret_flags, (ret_quads, ret_texts, *args[2:])

class ToTensor(object):
    def __call__(self, bboxes, labels, flags, *args):
        """
        :param bboxes:
        :param labels:
        :param flags:
        :param args:
            quads
            texts
        :return:
        """
        return torch.from_numpy(bboxes), torch.from_numpy(labels), flags, (torch.from_numpy(args[0]), *args[1:])


class ToQuadrilateral(object):
    def __call__(self, bboxes, labels, flags, *args):
        # Note that bboxes must be [cx, cy, w, h]
        assert args[0].shape[1] == 8, '4th arguments must be quadrilateral points'

        # b=(xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)
        # b's shape = (*, 4, 2=(top left, top right, bottom right, bottom left)=(x,y))
        box_num = bboxes.shape[0]
        b = np.zeros(shape=(box_num, 4, 2), dtype=np.float32)
        b[:, 0, :] = bboxes[:, :2]  # top left
        b[:, 1, :] = bboxes[:, np.array((2, 1))]
        b[:, 2, :] = bboxes[:, 2:]
        b[:, 3, :] = bboxes[:, np.array((0, 3))]

        # convert shape to (*, 4, 2)
        quads = args[0].reshape((-1, 4, 2))

        """
        dist formula is below;
        b[0] - q[0-3], b[1] - q[1-3,0], b[2] - q[2-3,0-1], b[3] - q[3,0-2]
        """

        dist = np.zeros(shape=(box_num, 4, 4))
        #trans = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
        trans = np.arange(4)
        for i in range(4):
            _b = np.expand_dims(b[:, i, :], axis=1)
            _q = quads[:, np.roll(trans, i), :]# shape = (?, 4, 2)
            dist[:, i, :] = np.linalg.norm(_b - _q, axis=-1)

        inds = np.argmin(dist.sum(axis=1), axis=-1)

        # update
        for b in range(box_num):
            quads[b] = quads[b, np.roll(trans, inds[b])]

        return bboxes, labels, flags, (quads.reshape((-1, 8)), *args[1:])

from ssd.core.inference import *
from .boxes.utils import quads_iou

import numpy as np
import cv2
import torch

def non_maximum_suppression_quads(pred, val_config):
    """
    :param pred: tensor, shape = (filtered default boxes num, 12=bbox+quad + 1=conf)
    :param val_config: SSDValConfig
    :return:
    """
    topk = val_config.topk
    iou_threshold2 = val_config.iou_threshold2

    loc, quad, conf = pred[:, :4], pred[:, 4:12], pred[:, -1]

    # sort confidence and default boxes with descending order
    c, conf_des_inds = conf.sort(dim=0, descending=True)
    # get topk indices
    conf_des_inds = conf_des_inds[:topk]

    inferred_indices = []
    while conf_des_inds.nelement() > 0:
        largest_conf_index = conf_des_inds[0]
        # conf[largest_conf_index]'s shape = []
        largest_conf = conf[largest_conf_index].unsqueeze(0).unsqueeze(0)  # shape = (1, 1)
        largest_conf_quad = quad[largest_conf_index, :].unsqueeze(0)  # shape = (1, 4=(xmin, ymin, xmax, ymax))
        # append to result
        inferred_indices.append(largest_conf_index)

        # remove largest element
        conf_des_inds = conf_des_inds[1:]

        if conf_des_inds.nelement() == 0:
            break

        # get iou, shape = (1, loc_des num)
        overlap = quads_iou(largest_conf_quad, quad[conf_des_inds])
        # filter out overlapped boxes for box with largest conf, shape = (loc_des num)
        indicator = overlap.reshape((overlap.nelement())) <= iou_threshold2

        conf_des_inds = conf_des_inds[indicator]

    inferred_indices = torch.Tensor(inferred_indices).long()
    return inferred_indices, conf[inferred_indices], torch.cat((loc[inferred_indices], quad[inferred_indices]), dim=1)


def textbox_non_maximum_suppression(pred, val_config):
    """
    :param pred: tensor, shape = (filtered default boxes num, 12=bbox+quad + 1=conf)
    Note that filtered default boxes number must be more than 1
    :param val_config: SSDValConfig
    :return:
        inferred_indices: Tensor, shape = (inferred box num,)
        inferred_confs: Tensor, shape = (inferred box num,)
        inferred_locs: Tensor, shape = (inferred box num, 4)
    """
    loc, quad, conf = pred[:, :4], pred[:, 4:12], pred[:, -1]

    indices, _, _ = non_maximum_suppression(torch.cat((loc, conf.unsqueeze(1)), dim=1), val_config)
    if indices.nelement() == 0:
        return indices, conf[indices], torch.cat((loc[indices], quad[indices]), dim=1)

    non_maximum_suppression_quads(pred[indices], val_config)


    return indices, conf[indices], torch.cat((loc[indices], quad[indices]), dim=1)


def toVisualizeQuadsRGBimg(img, poly_pts, thickness=2, rgb=(255, 0, 0), verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param poly_pts: list of Tensor, centered coordinates, shape = (box num, ?*2=(x1, y1, x2, y2,...)).
    :param thickness: int
    :param rgb: tuple of int, order is rgb and range is 0~255
    :param verbose: bool, whether to show information
    :return:
        img: RGB order
    """
    # convert (c, h, w) to (h, w, c)
    img = tensor2cvrgbimg(img, to8bit=True).copy()
    #cv2.imshow('a', img)
    #cv2.waitKey()
    # print(locs)
    poly_pts_mm = poly_pts.detach().numpy()

    h, w, c = img.shape

    if verbose:
        print(poly_pts)
    for bnum in range(poly_pts_mm.shape[0]):
        img = img.copy()
        pts = poly_pts_mm[bnum]
        pts[0::2] *= w
        pts[1::2] *= h
        pts[0::2] = np.clip(pts[0::2], 0, w)
        pts[1::2] = np.clip(pts[1::2], 0, h)
        #print(pts)
        pts = pts.reshape((-1, 1, 2)).astype(int)
        #print(pts)
        if verbose:
            print(pts)
        cv2.polylines(img, [pts], isClosed=True, color=rgb, thickness=thickness)

    return img

def toVisualizeInfQuadsRGBimg(img, poly_pts, inf_labels, classe_labels, inf_confs=None, tensor2cvimg=True, verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param poly_pts: list of Tensor, centered coordinates, shape = (box num, ?*2=(x1, y1, x2, y2,...)).
    :param inf_labels:
    :param classe_labels: list of str
    :param inf_confs: Tensor, (box_num,)
    :param tensor2cvimg: bool, whether to convert Tensor to cvimg
    :param verbose: bool, whether to show information
    :return:
        img: RGB order
    """
    # convert (c, h, w) to (h, w, c)
    if tensor2cvimg:
        img = tensor2cvrgbimg(img, to8bit=True).copy()
    else:
        if not isinstance(img, np.ndarray):
            raise ValueError('img must be Tensor, but got {}. if you pass \'Tensor\' img, set tensor2cvimg=True'.format(type(img).__name__))


    #cv2.imshow('a', img)
    #cv2.waitKey()
    # print(locs)
    poly_pts_mm = poly_pts.cpu().numpy()

    class_num = len(classe_labels)
    box_num = poly_pts.shape[0]
    assert box_num == inf_labels.shape[0], 'must be same boxes number'
    if inf_confs is not None:
        if isinstance(inf_confs, torch.Tensor):
            inf_confs = inf_confs.cpu().numpy()
        elif not isinstance(inf_confs, np.ndarray):
            raise ValueError(
                'Invalid \'inf_confs\' argment were passed. inf_confs must be ndarray or Tensor, but got {}'.format(
                    type(inf_confs).__name__))
        assert inf_confs.ndim == 1 and inf_confs.size == box_num, "Invalid inf_confs"

    # color
    angles = np.linspace(0, 255, class_num).astype(np.uint8)
    # print(angles.shape)
    hsvs = np.array((0, 255, 255))[np.newaxis, np.newaxis, :].astype(np.uint8)
    hsvs = np.repeat(hsvs, class_num, axis=0)
    # print(hsvs.shape)
    hsvs[:, 0, 0] += angles
    rgbs = cv2.cvtColor(hsvs, cv2.COLOR_HSV2RGB).astype(np.int)

    # Line thickness of 2 px
    thickness = 1


    h, w, c = img.shape

    if verbose:
        print(poly_pts)
    for bnum in range(poly_pts_mm.shape[0]):
        img = img.copy()
        pts = poly_pts_mm[bnum]
        pts[0::2] *= w
        pts[1::2] *= h
        pts[0::2] = np.clip(pts[0::2], 0, w)
        pts[1::2] = np.clip(pts[1::2], 0, h)
        #print(pts)
        pts = pts.reshape((-1, 1, 2)).astype(int)
        #print(pts)

        index = inf_labels[bnum].item()
        if math.isnan(index):
            continue
        index = int(index)

        rgb = tuple(rgbs[index, 0].tolist())

        if verbose:
            print(pts)
        cv2.polylines(img, [pts], isClosed=True, color=rgb, thickness=thickness)

    return img

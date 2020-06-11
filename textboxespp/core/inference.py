from ssd.core.inference import tensor2cvrgbimg, centroids2corners, toVisualizeRectangleRGBimg

import numpy as np
import cv2

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

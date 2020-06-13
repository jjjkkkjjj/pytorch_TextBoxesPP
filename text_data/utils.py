from ssd_data.utils import batch_ind_fn as batch_ind_fn_ssd
import torch

def batch_ind_fn_droptexts(batch):
    """
    :param batch:
    :return:
        imgs: Tensor, shape = (b, c, h, w)
        targets: list of Tensor, whose shape = (object box num, 4 + class num) including background
    """
    imgs, gts, texts = list(zip(*batch))

    return torch.stack(imgs), gts
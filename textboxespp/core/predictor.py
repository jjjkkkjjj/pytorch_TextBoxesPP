from ssd.core.predict import PredictorBase
import torch

class TextBoxPredictor(PredictorBase):

    def forward(self, locs, confs):
        """
        :param locs: list of Tensor, Tensor's shape is (batch, c, h, w)
        :param confs: list of Tensor, Tensor's shape is (batch, c, h, w)
        :return: predicts: localization and confidence Tensor, shape is (batch, total_dbox_num * (4=(cx,cy,w,h)+8=(x1,y1,x2,y2,...)+class_labels))
        """
        locs_reshaped, confs_reshaped = [], []
        for loc, conf in zip(locs, confs):
            batch_num = loc.shape[0]

            # original feature => (batch, (class_num or 4)*dboxnum, fmap_h, fmap_w)
            # converted into (batch, fmap_h, fmap_w, (class_num or 4)*dboxnum)
            # contiguous means aligning stored 1-d memory for given array
            loc = loc.permute((0, 2, 3, 1)).contiguous()
            locs_reshaped += [loc.reshape((batch_num, -1))]

            conf = conf.permute((0, 2, 3, 1)).contiguous()
            confs_reshaped += [conf.reshape((batch_num, -1))]



        locs_reshaped = torch.cat(locs_reshaped, dim=1).reshape((batch_num, -1, 12))
        confs_reshaped = torch.cat(confs_reshaped, dim=1).reshape((batch_num, -1, self.class_nums))

        return torch.cat((locs_reshaped, confs_reshaped), dim=2)
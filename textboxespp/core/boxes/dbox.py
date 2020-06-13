from ssd.core.boxes.dbox import *

class DBoxTextBoxOriginal(DBoxSSDOriginal):

    def build(self, feature_layers, classifier_source_names, localization_layers):
        super().build(feature_layers, classifier_source_names, localization_layers)

        self.dbox_num_per_fmap = [num*2 for num in self.dbox_num_per_fmap]
        return self

    @property
    def dbox_num_per_fpixel(self):
        return [len(aspect_ratio)*2*2 for aspect_ratio in self.aspect_ratios]

    def forward(self):
        dboxes = []

        # fsize = []
        # sk = []
        # sk_ = []

        # conv4_3 has different scale
        fmap_h, fmap_w = self.fmap_sizes[0]
        scale_k = self.scale_conv4_3
        scale_k_plus = self.scale_min
        ars = self.aspect_ratios[0]
        defaultboxes = self._make(fmap_w, fmap_h, scale_k, scale_k_plus, ars)[0]
        # get offset
        offset = (1.0 / fmap_h) / 2.0
        # create default boxes with vertical offset
        defaultboxes = np.repeat(defaultboxes, 2, axis=0)
        defaultboxes[::2, 1] += offset
        dboxes += [defaultboxes]

        # fsize += [fmap_h]
        # sk += [scale_k]
        # sk_ += [scale_k_plus]
        for k in range(1, self.fmap_num):
            fmap_h, fmap_w = self.fmap_sizes[k]
            scale_k = self.get_scale(k, m=self.fmap_num - 1)
            scale_k_plus = self.get_scale(k + 1, m=self.fmap_num - 1)
            ars = self.aspect_ratios[k]
            defaultboxes = self._make(fmap_w, fmap_h, scale_k, scale_k_plus, ars)[0]
            # get offset
            offset = (1.0 / fmap_h) / 2.0
            # create default boxes with vertical offset
            defaultboxes = np.repeat(defaultboxes, 2, axis=0)
            defaultboxes[::2, 1] += offset
            dboxes += [defaultboxes]
        """
            fsize += [fmap_h]
            sk += [scale_k]
            sk_ += [scale_k_plus]
        print(fsize, sk, sk_)
        """
        # print(dboxes)

        dboxes = np.concatenate(dboxes, axis=0)
        dboxes = torch.from_numpy(dboxes).float()

        # ret_features = torch.cat(ret_features, dim=1)
        if self.clip:
            dboxes = dboxes.clamp(min=0, max=1)

        return dboxes  # , ret_features

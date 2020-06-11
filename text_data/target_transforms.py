import logging
from ssd_data.target_transforms import Ignore as _Ignore
from ssd_data._utils import _check_ins

from ssd_data.target_transforms import *

class Ignore(_Ignore):
    supported_key = ['illegible']

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
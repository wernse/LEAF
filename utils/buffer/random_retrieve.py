from utils.buffer.buffer_utils import random_retrieve
import pandas as pd

class Random_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.num_retrieve = params.get('eps_mem_batch')

    def retrieve(self, buffer, **kwargs):
        excl_indices = None
        batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else self.num_retrieve
        return random_retrieve(buffer, batch_size, excl_indices=excl_indices)

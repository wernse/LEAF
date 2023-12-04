import numpy as np
import wandb

from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch

class Buffer(torch.nn.Module):
    def __init__(self, mem_size, params):
        super().__init__()
        self.current_index = 0
        self.tmp_current_index = 0
        self.n_seen_so_far = 0
        self.tmp_n_seen_so_far = 0
        self.device = "cuda"

        # define buffer
        buffer_size = mem_size

        # Modify for validation set
        # buffer_size = int(buffer_size * (1 + params.validation_split))
        # wandb.log({'total_buffer': buffer_size})
        self.buffer_size = buffer_size
        print('buffer has %d slots' % buffer_size)
        input_size = [3, 32, 32]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(-1))
        self.buffer_weights = np.array([])

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        # define update and retrieve method
        self.update_method = Reservoir_update(params) #
        self.retrieve_method = Random_retrieve(params) #eps_mem_batch

    def update(self, x, y,**kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def current_size(self):
        return len(self.buffer_label[self.buffer_label != -1])
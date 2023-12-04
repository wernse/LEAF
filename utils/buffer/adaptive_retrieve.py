import wandb

from utils.buffer.buffer_utils import random_retrieve
import numpy as np
import torch
import math

# Reset the seed?
class Adaptive_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.num_retrieve = params.eps_mem_batch

    def retrieve(self, buffer, **kwargs):
        excl_indices = kwargs.get('excl_indices')
        validation_size = kwargs.get('validation_size')
        if validation_size is not None:
            current_size = buffer.current_size()
            if current_size < buffer.buffer_size or validation_size is None:
                batch_size = 0
            else:
                batch_size = int(current_size * validation_size)
            return random_retrieve(buffer, batch_size, return_indices=True)[2]
        # 2. Based on the validation results, alter the distribution of selection
        batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else self.num_retrieve

        return random_retrieve(buffer, batch_size, excl_indices=excl_indices)


def random_retrieve(buffer, num_retrieve, excl_indices=None, return_indices=False):
    filled_indices = np.arange(buffer.current_index)
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    num_retrieve = min(num_retrieve, valid_indices.shape[0])
    indices = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, replace=False)).long()

    x = buffer.buffer_img[indices]

    y = buffer.buffer_label[indices]

    if return_indices:
        return x, y, indices
    else:
        return x, y


"""
def retrieve(self, buffer, **kwargs):
    sub_x, sub_y = random_retrieve(buffer, self.subsample)
    grad_dims = []
    for param in buffer.model.parameters():
        grad_dims.append(param.data.numel())
    grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
    model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
    if sub_x.size(0) > 0:
        with torch.no_grad():
            logits_pre = buffer.model.forward(sub_x)
            logits_post = model_temp.forward(sub_x)
            pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
            post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
            scores = post_loss - pre_loss
            big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
        return sub_x[big_ind], sub_y[big_ind]
    else:
        return sub_x, sub_y
"""
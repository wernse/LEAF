import torch
import numpy as np
from utils.utils import maybe_cuda
from collections import defaultdict
from collections import Counter
import random


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

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    sim = torch.mm(x1, x2.t())/(w1 * w2.t()).clamp(min=eps)
    return sim


def get_grad_vector(pp, grad_dims):
    """
        gather the gradients in one vector
        the node and the gradient are different, gradient is the loss direction change but the parameter change is based on learning rate
            - large differences (high loss) will have a larger gradient and larger change on the weight
        zero_grad removes all gradients
        loss.backwards calculates the gradient change for each node that uses gradient.
         - When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True and store them in parameter.grad attribute for every parameter.
        self.opt.step() updates each node based on the gradient change * the learning rate
        size: 1094750
    """
    grads = maybe_cuda(torch.Tensor(sum(grad_dims)))
    grads.fill_(0.0)
    cnt = 0
    # pp -> for every layer in the network
    # flatten the grads based on index i.e. layers: grad_dims: [540, 20] -> grads: [0-540, 540-560]
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


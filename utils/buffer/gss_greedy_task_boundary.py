from collections import deque

import torch
import torch.nn.functional as F
import pandas as pd
from utils.buffer.buffer_utils import get_grad_vector, cosine_similarity
from utils.utils import maybe_cuda
import wandb
import numpy as np
import math


class GSSGreedyUpdateTaskBoundary(object):
    def __init__(self, params):
        super().__init__()
        self.mem_strength = params.gss_mem_strength
        self.gss_batch_size = params.gss_batch_size
        self.buffer_score = maybe_cuda(torch.FloatTensor(params.mem_size).fill_(0))
        self.epsilon = float(params.epsilon)
        self.tasks = params.num_tasks
        self.policy = params.policy
        self.total_replacement_count = 0

    def update(self, buffer, x, y, **kwargs):
        t = kwargs.get('t')
        tasks_to_preserve = self.tasks

        wandb.log({'replacements': self.total_replacement_count})
        print("total replacements:", self.total_replacement_count)
        buffer.model.eval()

        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())

        place_left = buffer.buffer_img.size(0) - buffer.current_index
        if place_left <= 0:
            # Begin λSW by defining policies
            scale = 2
            slope = 1
            phase = tasks_to_preserve - 3
            policies = {
                'sig': lambda t: scale / (scale + np.exp(slope * (-t + phase))),
                'linear': lambda t: t / (tasks_to_preserve - 1),
                'exp': lambda t: 2 ** t / 2 ** (tasks_to_preserve - 1),
                'log': lambda t: math.log(t + 1) / math.log(tasks_to_preserve),
             }

            applied_policy = policies.get(self.policy)
            # If not policy, perform original GSS
            plasticity_policy = 0 if applied_policy is None else self.epsilon * applied_policy(t)

            batch_sim, mem_grads = self.get_batch_sim(buffer, grad_dims, x, y)

            policy_log = {
                't': t,
                'plasticity_policy': plasticity_policy,
                'epsilon': self.epsilon,
                'batch_sim': batch_sim,
            }

            wandb.log(policy_log)
            # Compare λ (batch_sim) and plasticity policy 
            if (batch_sim) + (plasticity_policy) < 0:
                # End λSW 
                buffer_score = self.buffer_score[:buffer.current_index].cpu()
                buffer_sim = (buffer_score - torch.min(buffer_score)) / \
                             ((torch.max(buffer_score) - torch.min(buffer_score)) + 0.01)
                index = torch.multinomial(buffer_sim, x.size(0), replacement=False)
                batch_item_sim = self.get_each_batch_sample_sim(buffer, grad_dims, mem_grads, x, y)
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(1)
                outcome = torch.multinomial(torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1), 1,
                                            replacement=False)
                added_indx = torch.arange(end=batch_item_sim.size(0))
                sub_index = outcome.squeeze(1).bool()
                buffer.buffer_img[index[sub_index]] = x[added_indx[sub_index]].clone()
                buffer.buffer_label[index[sub_index]] = y[added_indx[sub_index]].clone()
                df = pd.DataFrame(buffer.buffer_label.cpu())
                mem_items = {f"b_c_{k if k > 10 else f'i{k}'}":v for k,v in df[0].value_counts().to_dict().items()}
                # wandb.log(mem_items)
                self.buffer_score[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
                self.total_replacement_count += 1

        else:
            offset = min(place_left, x.size(0))
            x = x[:offset]
            y = y[:offset]
            if buffer.current_index == 0:
                batch_sample_memory_cos = torch.zeros(x.size(0)) + 0.1
            else:
                mem_grads = self.get_rand_mem_grads(buffer, grad_dims)
                batch_sample_memory_cos = self.get_each_batch_sample_sim(buffer, grad_dims, mem_grads, x, y)
            buffer.buffer_img[buffer.current_index:buffer.current_index + offset].data.copy_(x)
            buffer.buffer_label[buffer.current_index:buffer.current_index + offset].data.copy_(y)
            self.buffer_score[buffer.current_index:buffer.current_index + offset] \
                .data.copy_(batch_sample_memory_cos)
            buffer.current_index += offset
        buffer.model.train()

    def get_batch_sim(self, buffer, grad_dims, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            batch_x: batch images
            batch_y: batch labels
        Returns: score of current batch, gradient from memory subsets
        """
        mem_grads = self.get_rand_mem_grads(buffer, grad_dims)
        buffer.model.zero_grad()
        loss = F.cross_entropy(buffer.model.forward(batch_x), batch_y)
        loss.backward()
        batch_grad = get_grad_vector(buffer.model.parameters, grad_dims).unsqueeze(0)
        batch_sim = max(cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, buffer, grad_dims):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
        Returns: gradient from memory subsets
        """
        gss_batch_size = min(self.gss_batch_size, buffer.current_index)
        num_mem_subs = min(self.mem_strength, buffer.current_index // gss_batch_size)
        mem_grads = maybe_cuda(torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32))
        shuffeled_inds = torch.randperm(buffer.current_index)
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                                i * gss_batch_size:i * gss_batch_size + gss_batch_size]
            batch_x = buffer.buffer_img[random_batch_inds]
            batch_y = buffer.buffer_label[random_batch_inds]
            buffer.model.zero_grad()
            loss = F.cross_entropy(buffer.model.forward(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(get_grad_vector(buffer.model.parameters, grad_dims))
        return mem_grads

    def get_each_batch_sample_sim(self, buffer, grad_dims, mem_grads, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = maybe_cuda(torch.zeros(batch_x.size(0)))
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            buffer.model.zero_grad()
            ptloss = F.cross_entropy(buffer.model.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            this_grad = get_grad_vector(buffer.model.parameters, grad_dims).unsqueeze(0)
            cosine_sim[i] = max(cosine_similarity(mem_grads, this_grad))
            if torch.isnan(cosine_sim[i]):
                cosine_sim[i] = 0
        return cosine_sim

from random import uniform
import numpy as np
import torch
import pandas as pd
import wandb


class Reservoir_update(object):
    def __init__(self, params):
        super().__init__()

    def update(self, buffer, x, y, **kwargs):
        batch_size = x.size(0)
        # add whatever still fits in the buffer
        place_left = max(0, buffer.buffer_img.size(0) - buffer.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.buffer_img[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])
            weights = np.empty(offset)
            weights.fill(1)
            buffer.buffer_weights = np.append(buffer.buffer_weights, values=weights)

            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                filled_idx = list(range(buffer.current_index - offset, buffer.current_index, ))
                return filled_idx

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        # indices = indexes to replace
        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()
        valid_indices = (indices < buffer.buffer_img.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer  = indices[idx_new_data]
        # idx_buffer = indexes to replace
        buffer.n_seen_so_far += x.size(0)


        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < buffer.buffer_img.size(0)
        assert idx_buffer.max() < buffer.buffer_label.size(0)
        # assert idx_buffer.max() < self.buffer_task.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)
        # idx_map = indicies to index in
        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        replace_y = y[list(idx_map.values())]
        # perform overwrite op

        buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.buffer_label[list(idx_map.keys())] = replace_y
        buffer.buffer_weights[list([int(x) for x in idx_map.keys()])] = 1
        df = pd.DataFrame(buffer.buffer_label.cpu())
        mem_items = {f"b_c_{k if k > 10 else f'i{k}'}": v for k, v in df[0].value_counts().to_dict().items()}
        # wandb.log(mem_items)
        return list(idx_map.keys())
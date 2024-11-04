import numpy as np
import torch
import wandb


class Weighted_Reservoir_update(object):
    def __init__(self, params):
        super().__init__()
        self.tasks = params.num_tasks
        self.max_tasks = params.num_tasks

    def weighted_reservoir_sample(self, population, weights, k):
        """
        Args:
            population: list of indicies
            weights: list of weights
            k: number of samples returned

        Returns:
            a list of indices
        """
        # Get cumulative weights
        wc = np.cumsum(weights)
        # Total of weights
        m = wc[-1]
        sample = np.empty(k, population.dtype)
        sample_idx = np.full(k, -1, np.int32)
        # Sampling loop
        i = 0
        while i < k:
            # Pick random weight value
            r = m * np.random.rand()
            # Get corresponding index
            idx = np.searchsorted(wc, r, side='right')
            # Check index was not selected before
            if np.isin(idx, sample_idx):
                continue
            # Save sampled value and index
            sample[i] = population[idx]
            sample_idx[i] = population[idx]
            i += 1
        return sample

    def update(self, buffer, x, y, **kwargs):
        # Generate tmp buffer by altering the weights
        # Continue the original weights
        batch_size = x.size(0)
        place_left = max(0, buffer.buffer_img.size(0) - buffer.current_index)
        self.update_original_buffer(buffer, x, y)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.buffer_img[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])
            weights = np.empty(offset)
            weights.fill(1)
            buffer.buffer_weights = np.append(buffer.buffer_weights, values=weights)

            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            if offset == x.size(0):
                filled_idx = list(range(buffer.current_index - offset, buffer.current_index, ))
                return filled_idx
            assert all(buffer.buffer_weights)

        x, y = x[place_left:], y[place_left:]

        new_task = kwargs.get('new_task')
        lambda_param = kwargs.get('lambda_param')
        if new_task:
            buffer.buffer_img = buffer.tmp_buffer_img.detach().clone()
            buffer.buffer_label = buffer.tmp_buffer_label.detach().clone()
            buffer.current_index = buffer.tmp_current_index
            empty_weights = np.empty(buffer.tmp_current_index)
            empty_weights.fill(lambda_param)
            buffer.buffer_weights = empty_weights

        buffer.n_seen_so_far += x.size(0)
        population = np.arange(0, buffer.n_seen_so_far, 1)
        weights = np.empty(len(population) - len(buffer.buffer_weights))
        weights.fill(1)

        final_weights = np.append(buffer.buffer_weights, weights)
        k = x.size(0)

        indices = torch.FloatTensor(self.weighted_reservoir_sample(population, final_weights, k)).to(x.device)
        valid_indices = (indices < buffer.buffer_img.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}
        replace_y = y[list(idx_map.values())]

        buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.buffer_label[list(idx_map.keys())] = replace_y
        buffer.buffer_weights[list([int(x) for x in idx_map.keys()])] = 1

        return None

    def update_original_buffer(self, buffer, x, y):
        batch_size = x.size(0)
        place_left = max(0, buffer.tmp_buffer_img.size(0) - buffer.tmp_current_index)

        # Create
        if place_left:
            offset = min(place_left, batch_size)
            buffer.tmp_buffer_img[buffer.tmp_current_index: buffer.tmp_current_index + offset].data.copy_(x[:offset])
            buffer.tmp_buffer_label[buffer.tmp_current_index: buffer.tmp_current_index + offset].data.copy_(y[:offset])
            weights = np.empty(offset)
            weights.fill(1)
            buffer.tmp_buffer_weights = np.append(buffer.tmp_buffer_weights, values=weights)

            buffer.tmp_current_index += offset
            buffer.tmp_n_seen_so_far += offset

            if offset == x.size(0):
                filled_idx = list(range(buffer.tmp_current_index - offset, buffer.tmp_current_index, ))
                return filled_idx
            assert all(buffer.tmp_buffer_weights)

        x, y = x[place_left:], y[place_left:]

        buffer.tmp_n_seen_so_far += x.size(0)
        population = np.arange(0, buffer.tmp_n_seen_so_far, 1)
        weights = np.empty(len(population) - len(buffer.tmp_buffer_weights))
        weights.fill(1)
        final_weights = np.append(buffer.tmp_buffer_weights, weights)
        k = x.size(0)

        indices = torch.FloatTensor(self.weighted_reservoir_sample(population, final_weights, k)).to(x.device)
        valid_indices = (indices < buffer.tmp_buffer_img.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}
        replace_y = y[list(idx_map.values())]

        buffer.tmp_buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.tmp_buffer_label[list(idx_map.keys())] = replace_y
        buffer.tmp_buffer_weights[list([int(x) for x in idx_map.keys()])] = 1
        return list(idx_map.keys())

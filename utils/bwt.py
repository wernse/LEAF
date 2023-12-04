import numpy as np
import torch
from matplotlib import pyplot as plt
from copy import deepcopy

from networks.subnet2 import percentile
from utils.utils import count_grads_gt_zero


def print_output_head(model):
    last_task_weight = getattr(getattr(model, 'last')[11], 'weight') # 100x2048
    last_task_weight_grad = last_task_weight.grad # 100x2048

    conv1_grad = getattr(getattr(model, 'conv1'), 'weight').grad[0][0] # 100x2048
    fc_1_grad = getattr(getattr(model, 'fc1'), 'weight').grad[0]
    # getattr(getattr(model, 'conv1'),'weight').grad

    tensor_np = last_task_weight[0].cpu().detach().numpy()

    # Create an array of indices for x-axis
    indices = np.arange(len(tensor_np))

    plt.bar(indices, tensor_np)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Chart of Tensor Values')
    # plt.xlim(-1, 10)  # Set the x limit from 0 to 10
    plt.show()


def backwards_transfer(model, optimizer, criterion, task_id_nominal, buffer, training_masks, consolidated_masks, task_banana_id, finetune_last_layer_id):
    model.train()

    # fine-tune
    task_apple_mask_banana_task_id = finetune_last_layer_id
    banana_mask = training_masks[task_banana_id]
    for data, target in buffer[task_id_nominal]:
        optimizer.zero_grad()

        # If Mask then it will only use the current mask during training else will prune based on sparsity
        # Alexnet 799 -> Applies weight at every layer
        output = model(x=data,
                       task_id=task_apple_mask_banana_task_id,
                       mask=banana_mask,
                       mode="train")  # When training the mask is None. subnet.py forward

        loss = criterion(output, target)
        loss.backward()

        # Zero out frozen gradients in network
        if consolidated_masks is not None and consolidated_masks != {}:  # Only do this for tasks 1 and beyond
            for key in consolidated_masks.keys():
                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else:  # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

        optimizer.step()
    print(f"Finished fine tuning on task {task_id_nominal} with mask {task_banana_id}, apple output layer id {task_apple_mask_banana_task_id}")


def backwards_transfer_combined(model, optimizer, criterion, task_id_nominal,
                                buffer,
                                training_masks,
                                consolidated_masks,
                                finetune_last_layer_id,
                                last_layer_mask):
    model.train()

    # fine-tune
    index = 0
    for data, target in buffer[task_id_nominal]:
        optimizer.zero_grad()

        combined_last_layer = getattr(model, 'last')[finetune_last_layer_id]
        # If Mask then it will only use the current mask during training else will prune based on sparsity
        # Alexnet 799 -> Applies weight at every layer
        output = model(x=data,
                       task_id=finetune_last_layer_id,
                       training_masks=training_masks,
                       mode="train",
                       joint=True)  # When training the mask is None. subnet.py forward

        loss = criterion(output, target)
        loss.backward()

        # Zero out frozen gradients in network
        if consolidated_masks is not None and consolidated_masks != {}:  # Only do this for tasks 1 and beyond
            for key in consolidated_masks.keys():
                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else:  # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

        # Zero out last layer
        combined_last_layer = getattr(model, 'last')[finetune_last_layer_id]
        combined_last_layer.weight.grad[:, :2048][last_layer_mask == 1] = 0


        # if index == 0:
            # print(loss)
            # print(f"{count_grads_gt_zero(model)} last layer sparsity: {torch.sum(last_layer_mask == 0)} / {combined_last_layer.weight.view(-1).shape}")
            # print(f"cnt: {count_grads_gt_zero(model)}/ {combined_last_layer.weight.view(-1).shape}")
            # mask = combined_last_layer.weight[:, :2048] != 0
            # non_zero_gradients = combined_last_layer.weight.grad[:, :2048][mask]
            # avg_value_grad = torch.mean(non_zero_gradients.abs())
            # non_zero_gradients = combined_last_layer.weight[:, :2048][mask]
            # avg_value = torch.mean(non_zero_gradients.abs())
            # print(f"A Grads > 0: {avg_value_grad.item()}, A Weights > 0: {avg_value.item()}")
            #
            # mask = combined_last_layer.weight.grad[:, 2048:] != 0
            # non_zero_gradients = combined_last_layer.weight.grad[:, 2048:][mask]
            # avg_value_grad = torch.mean(non_zero_gradients.abs())
            # non_zero_gradients = combined_last_layer.weight[:, 2048:][mask]
            # avg_value = torch.mean(non_zero_gradients.abs())
            # print(f"B Grads > 0: {avg_value_grad.item()}, B Weights > 0: {avg_value.item()}")

        index += 1

        optimizer.step()
    print(f"Finished fine tuning on task {task_id_nominal} apple output layer id {finetune_last_layer_id}")


def joint_training(model, optimizer, criterion, task_id_nominal, buffer, training_masks, consolidated_masks):
    model.train()

    # fine-tune
    index = 0
    for data, target in buffer[task_id_nominal]:
        optimizer.zero_grad()

        # If Mask then it will only use the current mask during training else will prune based on sparsity
        output = model(x=data,
                       task_id=task_id_nominal,
                       training_masks=training_masks,
                       mode="train",
                       joint=True)

        loss = criterion(output, target)
        loss.backward()

        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if consolidated_masks is not None and consolidated_masks != {}:  # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():

                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else:  # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

        # count_grads_gt_zero(model)
        for name, param in model.named_parameters():
            if 'adapt' not in name and 'last' not in name:
                param.grad.zero_()
        if index == 0:
            print("cnt:", count_grads_gt_zero(model))
        index += 1
        optimizer.step()
    print(f"Finished fine tuning on task {task_id_nominal} with mask, {round(loss.item(),4)}")

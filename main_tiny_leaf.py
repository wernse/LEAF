import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import os
import os.path
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
from scipy.stats import wasserstein_distance

import argparse,time
import math
from copy import deepcopy, copy
from itertools import combinations, permutations

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.resnet18 import SubnetBasicBlock
from networks.tinynet import SubNet
from networks.utils import *
from utils import *
from utils import bwt as bwt_utils
from networks.lenet import SubnetLeNet as LeNet
from utils.utils import maybe_cuda
from networks.alexnet import SubnetAlexNet_NN as AlexNet, SubnetAlexNet_NN_LEAF
from networks.mlp import SubnetMLPNet as MLPNet
from networks.resnet18 import SubnetResNet18 as ResNet18

import importlib
import similarities
import wandb
# Test the amount of epochs 20 for main and 30 for FT
parent_bwt = []

batch_memory = []
def log_memory_usage():
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device_id)
    memory_utilization = torch.cuda.max_memory_allocated(device=None)
    batch_memory.append(torch.cuda.max_memory_allocated(device=None))
    torch.cuda.reset_peak_memory_stats(device=None)
    print(f'Memory Utilization: {memory_utilization / (1024 ** 2)} MB')
    import os
    import psutil

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # in bytes

    print(f"Memory Usage: {memory_usage / (1024 * 1024):.2f} MB")

# Placeholder for the function that will get the gradients
def get_gradients(x, y, device, model, criterion, task_id_nominal, mask):
    b = args.batch_size_train
    data = x[:b]
    data, target = data.to(device), y[:b].to(device)

    # Use model in evaluation mode to get gradients without dropout or batchnorm effects
    model.eval()
    data = data.requires_grad_(True)
    output = model(data, task_id_nominal, mask=mask, mode="train")
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    grad_dims = []
    for name, param in model.named_parameters():
        if 'last' in name or 'w_m' in name:
            continue
        grad_dims.append(param.data.numel())
    grads = maybe_cuda(torch.Tensor(sum(grad_dims)))
    grads.fill_(0.0)
    cnt = 0
    for name, param in model.named_parameters():
        if 'last' in name or 'w_m' in name:
            continue
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1

    return grads.cpu().numpy()
    # return data.grad.reshape(-1).cpu().numpy()


def train(args, model, device, train_loader, optimizer,criterion, task_id_nominal, consolidated_masks):
    model.train()

    total_loss = 0
    total_num = 0
    correct = 0

    # Loop batches
    for (k, (v_x, v_y)) in enumerate(train_loader):
        data = v_x.to(device)
        target = v_y.to(device)

        perm = torch.randperm(v_x.size(0))
        data = data[perm]
        target = target[perm]

        optimizer.zero_grad()
        output = model(data, task_id_nominal, mask=None, mode="train")

        loss = criterion(output, target)
        loss.backward()

        pred = output.argmax(dim=1, keepdim=True)

        correct    += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.data.cpu().numpy().item()*data.size(0)
        total_num  += data.size(0)

        # Continual Subnet no backprop
        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():
                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine wheter it's an output head or not
                key_split = key.split('.')
                if 'last' in key_split or len(key_split) == 2:
                    if 'last' in key_split:
                        module_attr = key_split[-1]
                        task_num = int(key_split[-2])
                        module_name = '.'.join(key_split[:-2])

                    else:
                        module_attr = key_split[1]
                        module_name = key_split[0]

                    # Zero-out gradients
                    if (hasattr(getattr(model, module_name), module_attr)):
                        if (getattr(getattr(model, module_name), module_attr) is not None):
                            getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

                else:
                    module_attr = key_split[-1]

                    # Zero-out gradients
                    curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])
                    if hasattr(curr_module, module_attr):
                        if getattr(curr_module, module_attr) is not None:
                            getattr(curr_module, module_attr).grad[consolidated_masks[key] == 1] = 0

        optimizer.step()

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def test(args, model, device, test_loader, criterion, task_id_nominal, curr_task_masks=None, mode="test"):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    with torch.no_grad():
        # Loop batches
        for (k, (v_x, v_y)) in enumerate(test_loader):
            data = v_x.to(device)
            target = v_y.to(device)
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*data.size(0)
            total_num  += data.size(0)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def eval_class_tasks(model, tasks, args, criterion, curr_task_masks=None, mode='test', idx=-1, device=None, end_idx=-1, comp_flag=False, encoding='huffman'):
    model.eval()

    result_acc = []
    result_lss = []
    comp_ratio = None

    with torch.no_grad():
        # Loop batches
        for t, task_loader in enumerate(tasks):

            if idx == -1 or idx == t:
                lss = 0.0
                acc = 0.0

                for (i, (x, y)) in enumerate(task_loader):
                    data = x.to(device)
                    target = y.to(device)

                    if curr_task_masks is not None:
                        if comp_flag:
                            if encoding == 'huffman':
                                per_task_mask, comp_ratio = comp_decomp_mask_huffman(curr_task_masks, t, device)
                            else:
                                per_task_mask, comp_ratio = comp_decomp_mask(curr_task_masks, t, device)
                            output = model(data, t, mask=per_task_mask, mode=mode)
                        else:
                            output = model(data, t, mask=curr_task_masks[t], mode=mode)
                    else:
                        output = model(data, t, mask=None, mode=mode)
                    loss = criterion(output, target)

                    pred = output.argmax(dim=1, keepdim=True).detach()
                    acc += pred.eq(target.view_as(pred)).sum().item()
                    lss += loss.data.cpu().numpy().item()*data.size(0)

                    _, p = torch.max(output.data.cpu(), 1, keepdim=False)

                result_lss.append(lss / len(task_loader.dataset))
                result_acc.append(acc / len(task_loader.dataset))

    return result_lss[-1], result_acc[-1] * 100, comp_ratio


def main(args):
    tstart=time.time()
    ## Device Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Loader = importlib.import_module('dataloader.' + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks, input_size = loader.get_dataset_info()
    num_tasks = loader.n_tasks
    model = SubNet(input_size, n_outputs, n_tasks, args.sparsity).to(device)

    # input_size: ch * size * size = n_inputs
    print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
    print('-' * 100)


    acc_matrix=np.zeros((n_tasks,n_tasks))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()
    # Load test and val datasets
    test_tasks = loader.get_tasks('test')
    val_tasks = loader.get_tasks('val')
    org_valid_loss_list = []
    org_train_loss_list = []
    org_valid_acc_list = []
    distances = {}
    margin = 0  # Allowance parameter, typically half the shift to be detected
    h = args.threshold  # Decision interval or threshold for signaling a change, depends on system tolerance

    # Initialize the CUSUM variables
    S_pos = 0.0
    S_neg = 0.0
    leaf_triggers = 0
    change_points = []
    consolidated_distances = {}
    cum_distances = []
    s_pos_list = []
    s_neg_list = []
    A_acc_list = []
    AB_acc_list = {}
    valid_loss_list = {}
    train_loss_list = {}
    valid_acc_list = {}
    mask_sim_list = {}
    task_list = []
    per_task_masks, consolidated_masks, per_int_masks, int_masks = {}, {}, {}, {}
    for k in range(num_tasks):
        loader.set_new_task(k)
        task_info, train_loader, _, _ = loader.new_task()
        task_id = task_info['task']
        print('*' * 100)
        print(f'Task {task_id:2d} {task_info.get("n_train_data")}')
        print('*' * 100)
        task_list.append(task_id)

        lr = args.lr
        best_loss = np.inf
        best_acc = 0
        best_mask = None
        print('-' * 40)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print('-' * 40)

        best_model = get_model(model)
        if args.optim == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")


        for epoch in range(1, args.n_epochs + 1):
            # Train
            clock0 = time.time()
            tr_loss, tr_acc = train(args, model, device, train_loader, optimizer, criterion, task_id,
                                    consolidated_masks)
            tmp_mask = model.get_masks(task_id)
            clock1 = time.time()
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, \
                                                                                             tr_loss, tr_acc,
                                                                                             1000 * (clock1 - clock0)),
                  end='')
            # Valid
            valid_loss, valid_acc, comp_ratio = eval_class_tasks(model=model,
                                                                 tasks=val_tasks,
                                                                 args=args,
                                                                 criterion=criterion,
                                                                 curr_task_masks=None, mode='valid', idx=task_id,
                                                                 device=device)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
            # Adapt lr
            if valid_acc > best_acc:
                best_loss = valid_loss
                best_acc = valid_acc
                best_model = get_model(model)
                best_mask = tmp_mask
                patience = args.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= args.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < args.lr_min:
                        print()
                        break
                    patience = args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)
            print()

        # Restore best model
        set_model_(model, best_model)
        org_valid_loss_list.append(round(best_loss, 4))
        org_train_loss_list.append(round(tr_loss, 4))
        org_valid_acc_list.append(round(best_acc, 4))

        # Save the per-task-dependent masks
        per_task_masks[task_id] = best_mask

        # Consolidate task masks to keep track of parameters to-update or not
        if task_id == 0:
            consolidated_masks = deepcopy(per_task_masks[task_id])
        else:
            for key in per_task_masks[task_id].keys():
                if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                    consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))

        # Print Sparsity
        sparsity_per_layer = print_sparsity(consolidated_masks)
        all_sparsity = global_sparsity(consolidated_masks)
        print("Global Sparsity: {}".format(all_sparsity))
        sparsity_matrix.append(all_sparsity)
        sparsity_per_task[task_id] = sparsity_per_layer

        # Test
        print ('-'*40)
        test_loss, test_acc = test(args, model, device, test_tasks[task_id], criterion, task_id,
                                   curr_task_masks=best_mask, mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
        A_acc_list.append(round(test_acc, 4))

        #### LEAF ####
        target_backwards_transfer_task_id = args.target_backwards_transfer_task_id
        print(target_backwards_transfer_task_id)
        backwards_transfer_task_id = target_backwards_transfer_task_id

        if task_id > target_backwards_transfer_task_id:
            lr = args.lr

            consolidated_masks_B = {}
            print(f"Backwards Transfer from Task {task_id} to Task {backwards_transfer_task_id}")
            for existing_task_id in range(task_id + 1):
                if existing_task_id == backwards_transfer_task_id:
                    print("skipping frozen id", existing_task_id)
                    continue
                print("Adding recreating:", existing_task_id)
                if consolidated_masks_B == {}:
                    consolidated_masks_B = deepcopy(per_task_masks[existing_task_id])
                for key in per_task_masks[existing_task_id].keys():
                    if consolidated_masks_B[key] is not None and per_task_masks[existing_task_id][key] is not None:
                        consolidated_masks_B[key] = 1 - (
                                (1 - consolidated_masks_B[key]) * (1 - per_task_masks[existing_task_id][key]))

            loader.set_new_task(target_backwards_transfer_task_id)
            task_info, train_loader, _, _ = loader.new_task(seed=args.seed)
            A_train = train_loader

            for (k, (v_x, v_y)) in enumerate(val_tasks[backwards_transfer_task_id]):
                x = v_x.to(device)
                y = v_y.to(device)

            task_0 = get_gradients(x=x, y=y, model=model, device=device, criterion=criterion,
                                   task_id_nominal=target_backwards_transfer_task_id,
                                   mask=per_task_masks[target_backwards_transfer_task_id])
            task_1 = get_gradients(x=x, y=y, model=model, device=device, criterion=criterion,
                                   task_id_nominal=target_backwards_transfer_task_id, mask=per_task_masks[task_id])
            consolidated = get_gradients(x=x, y=y, model=model, device=device, criterion=criterion,
                                         task_id_nominal=target_backwards_transfer_task_id, mask=consolidated_masks_B)
            distance = wasserstein_distance(task_0, task_1)
            c_distance = wasserstein_distance(task_0, consolidated)
            print(f"t0->t{distance * 1000}, t0->c{c_distance * 1000}")
            distances.setdefault(leaf_triggers, []).append(round(distance * 1000, 4))
            consolidated_distances.setdefault(leaf_triggers, []).append(round(c_distance * 1000, 4))
            c_distance = c_distance * 1000
            target = 0 if len(cum_distances) < 0 else np.mean(cum_distances[:5])
            cum_distances.append(c_distance)
            # CUSUM Ωalgorithm
            S_pos = max(0, S_pos + c_distance - target - margin)
            S_neg = min(0, S_neg + c_distance - target + margin)
            s_pos_list.append(S_pos)
            s_neg_list.append(S_neg)
            if S_pos > h or S_neg < -h:
                change_points.append(task_id)
                # Reset the CUSUM statistics after a change is detected
                S_pos = 0.0
                S_neg = 0.0
                leaf_triggers += 1

                lr = args.lr
                best_loss = np.inf
                best_acc = 0
                best_mask = None
                print("breakpoint, testing the mask consolidation to fine-tune the first task with buffer size")

                optimizer_B = optim.Adam(model.parameters(), lr=lr)

                for epoch in range(1, args.n_epochs_finetuning):
                    clock0 = time.time()
                    tr_loss, tr_acc = train(args, model, device, A_train, optimizer_B, criterion, backwards_transfer_task_id,
                                            consolidated_masks)
                    clock1 = time.time()
                    tmp_mask = model.get_masks(backwards_transfer_task_id)
                    print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, \
                                                                                                     tr_loss, tr_acc,
                                                                                                     1000 * (
                                                                                                                 clock1 - clock0)),
                          end='')
                    # Valid
                    valid_loss, valid_acc = test(args, model, device, val_tasks[backwards_transfer_task_id], criterion, backwards_transfer_task_id,
                                               curr_task_masks=tmp_mask, mode="test")
                    print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
                    # Adapt lr
                    if valid_acc > best_acc:
                        best_loss = valid_loss
                        best_acc = valid_acc
                        best_model = get_model(model)
                        best_mask = tmp_mask
                        patience = args.lr_patience
                        print(' *', end='')
                    else:
                        patience -= 1
                        if patience <= 0:
                            lr /= args.lr_factor
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr < args.lr_min:
                                print()
                                break
                            patience = args.lr_patience
                            adjust_learning_rate(optimizer_B, epoch, args)
                    print()
                # log_memory_usage()
                set_model_(model, best_model)
                valid_loss_list.setdefault(str(backwards_transfer_task_id), []).append(round(best_loss, 4))
                train_loss_list.setdefault(str(backwards_transfer_task_id), []).append(round(tr_loss, 4))
                valid_acc_list.setdefault(str(backwards_transfer_task_id), []).append(best_acc)
                test_loss, test_acc = test(args, model, device, test_tasks[target_backwards_transfer_task_id], criterion, target_backwards_transfer_task_id, curr_task_masks=best_mask, mode="test")
                print(' Test: loss={:.3f}, acc={:5.1f}%'.format(test_loss, test_acc))
                AB_acc_list.setdefault(str(backwards_transfer_task_id), []).append(test_acc)

        print("AB_acc_list", AB_acc_list)
        print("A_acc_list", A_acc_list)
        print("org_train_loss_list", org_train_loss_list)
        print("train_loss_list", train_loss_list)
        print("org_valid_loss_list", org_valid_loss_list)
        print("valid_loss_list", valid_loss_list)
        print("org_valid_acc_list", org_valid_acc_list)
        print("valid_acc_list", valid_acc_list)
        print("batch_memory", batch_memory)
        print("distances", distances)
        print("consolidated_distances", consolidated_distances)
        print("cum_distances", cum_distances)
        print("s_pos_list", s_pos_list)
        print("s_neg_list", s_neg_list)
        wandb.log({'AB_acc_list': AB_acc_list})
        wandb.log({'A_acc_list': A_acc_list})
        wandb.log({'org_valid_loss_list': org_valid_loss_list})
        wandb.log({'valid_loss_list': valid_loss_list})
        wandb.log({'org_train_loss_list': org_train_loss_list})
        wandb.log({'train_loss_list': train_loss_list})
        wandb.log({'org_valid_acc_list': org_valid_acc_list})
        wandb.log({'valid_acc_list': valid_acc_list})
        wandb.log({'sparsity_matrix': sparsity_matrix})
        wandb.log({'batch_memory': batch_memory})
        wandb.log({'distances': distances})
        wandb.log({'consolidated_distances': consolidated_distances})
        wandb.log({'cum_distances': cum_distances})
        wandb.log({'s_pos_list': s_pos_list})
        wandb.log({'s_neg_list': s_neg_list})
        # update task id
        task_id +=1
    #

    # checkpoint = torch.load(f'checkpoints/{args.name}')


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid', default=0.1, type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")

    # Model parameters
    parser.add_argument('--model', type=str, default="tinynet", metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # data loader
    parser.add_argument('--loader', type=str,
                        default="class_incremental_loader",
                        metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')
    # workers
    parser.add_argument('--workers', type=int, default=0, metavar='S',
                        help='(default: 8)')
    # class order
    parser.add_argument('--class_order', type=str, default="random", metavar='MODEL',
                        help="")
    # dataset
    parser.add_argument('--dataset', type=str, default="tinyimagenet", metavar='',
                        help="")
    # dataset
    parser.add_argument('--data_path', type=str, default="data/tiny-imagenet-200/", metavar='',
                        help="")
    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")
    parser.add_argument('--memories', type=int, default=1000,
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--mem_batch_size', type=int, default=300,
                        help='the amount of items selected to update feature spaces.')

    # parser.add_argument('--use_track', type=str2bool, default=False)
    parser.add_argument('--freeze_bn', default=False, action='store_true', help='')

    parser.add_argument('--encoding', type=str, default="huffman", metavar='',
                        help="")

    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--soft', type=float, default=0.0)
    parser.add_argument('--soft_grad', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=6.0)
    parser.add_argument('--target_backwards_transfer_task_id', type=int, default=0)

    parser.add_argument('--n_epochs_finetuning', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')

    args = parser.parse_args()
    args.sparsity = 1 - args.sparsity

    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    # update name
    name = f"{args.name}_{args.dataset}_SEED_{args.seed}_n_epochs_{args.n_epochs}"
    args.name = name
    os.environ["WANDB_API_KEY"] = "YOUR KEY"
    wandb.init(project=f"LEAF_{args.dataset}", name=name, config=args)
    main(args)





import setproctitle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
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
import argparse,time
import math
from copy import deepcopy, copy
from itertools import combinations, permutations
#
from networks.subnet import SubnetLinear, SubnetConv2d
from networks.resnet18 import SubnetBasicBlock
from networks.utils import *
from utils import *
from utils import bwt as bwt_utils
from networks.lenet import SubnetLeNet as LeNet
from networks.alexnet import SubnetAlexNet_NN_overlap as AlexNet, SubnetAlexNet_NN_LEAF
from networks.mlp import SubnetMLPNet as MLPNet
from networks.resnet18 import SubnetResNet18 as ResNet18

import importlib
import similarities
import wandb
# Test the amount of epochs 20 for main and 30 for FT
parent_bwt = []


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


def train(args, model, device, x, y, optimizer, criterion, task_id_nominal, consolidated_masks, weight_overlap):
    """
     weight overlap = % of penalty for consolidated mask weights, when selecting w_m threshold
    """
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for idx, i in enumerate(range(0,len(r),args.batch_size_train)):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        try:
            output = model(data, task_id_nominal, mask=None, mode="train", consolidated_masks=consolidated_masks, weight_overlap=weight_overlap)
        except:
            output = model(data, task_id_nominal, mask=None, mode="train")
        loss = criterion(output, target)
        loss.backward()

        # Continual Subnet no backprop
        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if consolidated_masks is not None and consolidated_masks != {}:  # Only do this for tasks 1 and beyond
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


def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test", epoch=1):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if ((i + args.batch_size_test) <= len(r)):
                b = r[i:i + args.batch_size_test]
            else:
                b = r[i:]

            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc, None, None, None

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

    exp_dir = "results_{}".format(args.dataset)

    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
        np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
        np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
        np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])
    ]

    ## Load CIFAR100_SUPERCLASS DATASET
    ids = task_order[args.t_order]
    if args.dataset == 'cifar100_100':
        dataloader = importlib.import_module('dataloader.' + args.dataset)
        data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, args=args,
                                                                       seed=args.seed, pc_valid=args.pc_valid,
                                                                       samples_per_task=args.samples_per_task)
        print('Task info =', output_info)
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('Task order =', ids)
        print('-' * 100)

        tasks_ = [t for t in ids]
        n_outputs_ = [n_outputs] * n_tasks
        taskcla = [(t, n) for t, n in zip(tasks_, n_outputs_)]
        model = AlexNet(taskcla, args.sparsity).to(device)
    if args.dataset == 'cifar100_superclass100':
        dataloader = importlib.import_module('dataloader.' + args.dataset)
        data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, task_order=ids,
                                                                   seed=args.seed, pc_valid=args.pc_valid)
        print('Task info =', output_info)
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('Task order =', ids)
        print('-' * 100)

        tasks_ = [t for t in ids]
        n_outputs_ = [n_outputs] * n_tasks
        taskcla = [(t, n) for t, n in zip(tasks_, n_outputs_)]
        model = AlexNet(taskcla, args.sparsity).to(device)

    if args.dataset == 'omniglot':
        from networks.lenet import SubnetLeNet_Large as MLPNet
        from dataloader import omniglot_rotation
        data, taskcla, inputsize = omniglot_rotation.get(seed=args.seed, pc_valid=args.pc_valid)
        print(taskcla)
        # Model Instantiation
        model = MLPNet(taskcla, args.sparsity).to(device)
        n_tasks = len(taskcla)
    if args.dataset == 'five_data':
        from dataloader import five_datasets
        data, taskcla, inputsize = five_datasets.get(seed=args.seed, pc_valid=args.pc_valid)
        data_tmp = {}
        model = ResNet18(taskcla, nf=20, sparsity=args.sparsity).to(device) # base filters: 20
        dir = '5_data_checkpoints'
        n_tasks = len(taskcla)
    if args.dataset == 'miniimagenet':
        from dataloader import miniimagenet as data_loader
        dataloader = data_loader.DatasetGen(args)
        taskcla, input_size = dataloader.taskcla, dataloader.inputsize
        n_tasks = 20
        args.n_tasks = 20
        taskcla = [(x[0],100) for x in taskcla]
        model = ResNet18(taskcla, nf=20, sparsity=args.sparsity, size=input_size).to(device) # base filters: 20
        for x in range(20):
            dataloader.get(x)


    acc_matrix=np.zeros((n_tasks,n_tasks))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)
    _buffer = {}
    task_id = 0
    leaf_triggers = 0
    task_list = []
    org_valid_loss_list = []
    org_train_loss_list = []
    org_valid_acc_list = []
    A_acc_list = []
    AB_acc_list = {}
    distances = {}
    consolidated_distances = {}
    cum_distances = []
    s_pos_list = []
    s_neg_list = []

    valid_loss_list = {}
    train_loss_list = {}
    valid_acc_list = {}
    # CUSUM parameters
    margin = 0  # Allowance parameter, typically half the shift to be detected
    h = args.threshold # Decision interval or threshold for signaling a change, depends on system tolerance

    # Initialize the CUSUM variables
    S_pos = 0.0
    S_neg = 0.0
    change_points = []
    per_task_masks, consolidated_masks, per_prime_masks, prime_masks = {}, {}, {}, {}
    for k,ncla in taskcla:
        print('*'*100)
        print('*'*100)

        if args.dataset == 'miniimagenet':
            data = dataloader.get(k)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']

        task_list.append(k)

        lr = args.lr
        best_loss=np.inf
        best_acc=0
        best_mask=None
        print ('-'*40)
        print('Task ID :{} | Learning Rate : {}, | Shape : {}'.format(task_id, lr, ytrain.shape))
        print ('-'*40)

        best_model=get_model(model)
        if args.optim == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")

        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()
            train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks, weight_overlap=args.overlap_base)
            clock1 = time.time()
            tmp_mask = model.get_masks(task_id)
            tr_loss,tr_acc, tr_comp_ratio, prediction, true_y = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=tmp_mask, mode="valid")
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc,vr_comp_ratio, prediction, true_y = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=tmp_mask, mode="valid")
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            # Adapt lr
            if valid_acc > best_acc:
                best_loss=valid_loss
                best_acc=valid_acc
                best_model=get_model(model)
                best_mask=tmp_mask
                patience=args.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print()
                        break
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)
            print()

        # Restore the best model
        set_model_(model,best_model)
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
        test_loss, test_acc, test_comp_ratio, prediction, true_y = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
        A_acc_list.append(round(test_acc, 4))

        #### LEAF ####
        target_backwards_transfer_task_id = args.target_backwards_transfer_task_id
        backwards_transfer_task_id = taskcla[target_backwards_transfer_task_id][0]
        if task_id > target_backwards_transfer_task_id:
            print("reinit optimizer")
            lr = args.lr
            if args.optim == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif args.optim == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:
                raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")

            consolidated_masks_B = {}
            # 1.1 Load the consolidated_masks_B without backwards_transfer_task_id
            # task
            print(f"Backwards Transfer from Task {task_id} to Task {backwards_transfer_task_id}")
            for existing_task_id in range(task_id + 1):
                existing_task_id = taskcla[existing_task_id][0]
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

            task0_train_x = data[backwards_transfer_task_id]['train']['x']
            task0_train_y = data[backwards_transfer_task_id]['train']['y']

            task0_test_x = data[backwards_transfer_task_id]['test']['x']
            task0_test_y = data[backwards_transfer_task_id]['test']['y']
            task0_valid_x = data[backwards_transfer_task_id]['valid']['x']
            task0_valid_y = data[backwards_transfer_task_id]['valid']['y']

            max_size = int(len(task0_train_x) * args.buffer_percentage)
            task0_train_x = task0_train_x[:max_size]
            task0_train_y = task0_train_y[:max_size]
            x = task0_train_x
            y = task0_train_y

            task_0 = get_gradients(x=x, y=y, model=model, device=device, criterion=criterion,
                                   task_id_nominal=target_backwards_transfer_task_id,
                                   mask=per_task_masks[target_backwards_transfer_task_id])
            task_1 = get_gradients(x=x, y=y, model=model, device=device, criterion=criterion,
                                   task_id_nominal=target_backwards_transfer_task_id, mask=per_task_masks[task_id])
            consolidated = get_gradients(x=x, y=y, model=model, device=device, criterion=criterion,
                                         task_id_nominal=target_backwards_transfer_task_id, mask=consolidated_masks_B)
            distance = wasserstein_distance(task_0, task_1)
            c_distance = wasserstein_distance(task_0, consolidated)

            distances.setdefault(leaf_triggers, []).append(distance)
            consolidated_distances.setdefault(leaf_triggers, []).append(c_distance)

            target = 0 if len(cum_distances) < 0 else np.mean(cum_distances[:5])
            cum_distances.append(c_distance)

            # If change detected
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
                for epoch in range(1, args.n_epochs_finetuning):
                    clock0 = time.time()
                    train(args, model, device, task0_train_x, task0_train_y, optimizer, criterion,
                          backwards_transfer_task_id, consolidated_masks_B, weight_overlap=args.overlap)
                    clock1 = time.time()

                    tmp_mask = model.get_masks(backwards_transfer_task_id)
                    tr_loss, tr_acc, tr_comp_ratio, prediction, true_y = test(args, model, device, task0_train_x,
                                                                              task0_train_y,
                                                                              criterion, backwards_transfer_task_id,
                                                                              curr_task_masks=tmp_mask, mode="valid")
                    print(
                        f'Epoch {epoch} | Train: loss={round(tr_loss, 3)}, acc={round(tr_acc, 2)}% | time={round(1000 * (clock1 - clock0))}ms',
                        end='')
                    valid_loss, valid_acc, vr_comp_ratio, prediction, true_y = test(args, model, device, task0_valid_x,
                                                                                    task0_valid_y,
                                                                                    criterion, backwards_transfer_task_id,
                                                                                    curr_task_masks=tmp_mask,
                                                                                    mode="valid")
                    print(f' Valid: loss={round(valid_loss, 3)}, acc={round(valid_acc, 2)}% |', end='')

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

                set_model_(model, best_model)
                valid_loss_list.setdefault(str(backwards_transfer_task_id), []).append(round(best_loss, 4))
                train_loss_list.setdefault(str(backwards_transfer_task_id), []).append(round(tr_loss, 4))
                valid_acc_list.setdefault(str(backwards_transfer_task_id), []).append(best_acc)
                test_loss, test_acc, vr_comp_ratio, prediction, true_y = test(args, model, device, task0_test_x,
                                                                              task0_test_y,
                                                                              criterion,
                                                                              backwards_transfer_task_id,
                                                                              curr_task_masks=best_mask,
                                                                              mode="test")
                AB_acc_list.setdefault(str(backwards_transfer_task_id), []).append(test_acc)
                # Save the per-task-dependent masks
                per_task_masks[target_backwards_transfer_task_id] = best_mask

                # Consolidate task masks to keep track of parameters to-update or not
                for key in per_task_masks[target_backwards_transfer_task_id].keys():
                    if consolidated_masks_B[key] is not None and per_task_masks[target_backwards_transfer_task_id][key] is not None:
                        consolidated_masks_B[key] = 1 - (
                                    (1 - consolidated_masks_B[key]) * (1 - per_task_masks[target_backwards_transfer_task_id][key]))
                consolidated_masks = consolidated_masks_B

        # save accuracy
        for jj in range(n_tasks):

            xtest =data[jj]['test']['x']
            ytest =data[jj]['test']['y']

            if jj <= task_id:
                _, acc_matrix[task_id,jj], comp_ratio, prediction, true_y = test(args, model, device, xtest, ytest, criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            else:
                _, acc_matrix[task_id,jj], comp_ratio, prediction, true_y = test(args, model, device, xtest, ytest, criterion, jj, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(n_tasks):
                print('{:5.1f} '.format(acc_matrix[i_a, j_a]),end='')
            print()

        print("AB_acc_list", AB_acc_list)
        print("A_acc_list", A_acc_list)
        print("org_train_loss_list", org_train_loss_list)
        print("train_loss_list", train_loss_list)
        print("org_valid_loss_list", org_valid_loss_list)
        print("valid_loss_list", valid_loss_list)
        print("org_valid_acc_list", org_valid_acc_list)
        print("valid_acc_list", valid_acc_list)
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
        wandb.log({'distances': distances})
        wandb.log({'consolidated_distances': consolidated_distances})
        wandb.log({'cum_distances': cum_distances})
        wandb.log({'s_pos_list': s_pos_list})
        wandb.log({'s_neg_list': s_neg_list})

        task_id +=1

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')

    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="sgd", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')

    # Model parameters
    parser.add_argument('--model', type=str, default="resnet18", metavar='MODEL',
                        help="Models to be incorporated for the experiment")

    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")

    # miniImagenet parameters
    parser.add_argument('--nperm', type=int, default=20, metavar='NPERM',
                        help='number of permutations/tasks')

    # data parameters
    parser.add_argument('--loader', type=str,
                        default='task_incremental_loader',
                        help='data loader to use')
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')

    parser.add_argument('--data_path', default='./data/', help='path where data is located')
    parser.add_argument("--dataset",
                        default='mnist_permutations',
                        type=str,
                        required=True,
                        choices=['mnist_permutations', 'miniimagenet', 'cifar100_100', 'cifar100_superclass100', 'tinyimagenet', 'pmnist', 'five_data', 'omniglot'],
                        help="Dataset to train and test on.")


    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')

    parser.add_argument("--workers", default=4, type=int, help="Number of workers preprocessing the data.")

    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")
    parser.add_argument("--class_order", default="random", type=str, choices=["random", "chrono", "old", "super"],
                        help="define classes order of increment ")

    # For cifar100
    parser.add_argument('--n_tasks', type=int, default=20,
                        help='total number of tasks, invalid for cifar100_superclass')
    parser.add_argument('--shuffle_task', default=False, action='store_true',
                        help='Invalid for cifar100_superclass')
    parser.add_argument('--encoding', type=str, default="huffman", metavar='',
                        help="")

    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--soft', type=float, default=0.0)
    parser.add_argument('--soft_grad', type=float, default=1.0)
    parser.add_argument('--overlap', type=float, default=1.0)
    parser.add_argument('--overlap_base', type=float, default=1.0)
    parser.add_argument('--buffer_percentage', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=6)
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
    name = f"{args.name}_{args.dataset}_SEED_{args.seed}_n_epochs_{args.n_epochs}_pc_valid_{args.pc_valid}"
    args.name = name
    process_name = f"LEAF_{args.dataset}"
    setproctitle.setproctitle(process_name)
    os.environ["WANDB_API_KEY"] = "YOUR KEY"
    wandb.init(project=process_name, name=name, config=args)
    main(args)





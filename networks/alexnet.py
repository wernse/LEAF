import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from .subnet import SubnetConv2d, SubnetLinear
from .structured_subnet import StructuredSubnetConv2d, StructuredSubnetLinear
from .structured_subnet_v2 import StructuredSubnetConv2d as StructuredSubnetConv2d_v2
from .structured_subnet_v2 import StructuredSubnetLinear as StructuredSubnetLinear_v2
from .conditional_task import TaskLinear, TaskConv2d
## Define AlexNet model
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class STLAlexNet(nn.Module):
    def __init__(self, taskcla, sparsity):
        super(STLAlexNet, self).__init__()
        self.in_channel = []
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)

        self.taskcla = taskcla
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(2048, n, bias=False))

    def forward(self, x, task_id):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.reshape(bsz, -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = self.last[task_id](x)
        return y
# x.shape
# Out[3]: torch.Size([10, 256, 2, 2])
# bsz
# Out[4]: 10

# Define the multi-task neural network model
class MultiTaskModel(nn.Module):
    def __init__(self, taskcla):
        super(MultiTaskModel, self).__init__()
        self.relu = torch.nn.ReLU()

        self.in_channel = []
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

    def forward(self, x, task):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.view(x.size(0), 256 * 6 * 6)  # Flatten the tensor
        x_shared = self.relu(self.fc1_shared(x))

        if task == 'A':
            # Task A specific layers
            x_A = self.relu(self.fc2_A(x_shared))
            x_A = self.fc3_A(x_A)
            return x_A

        elif task == 'B':
            # Task B specific layers
            x_B = self.relu(self.fc2_B(x_shared))
            x_B = self.fc3_B(x_B)
            return x_B


class GPMAlexNet(nn.Module):
    def __init__(self, taskcla):
        super(GPMAlexNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048, n, bias=False))

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv2'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv3'] = x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.drop2(self.relu(x))
        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y


class CTAlexNet(nn.Module):
    def __init__(self, taskcla):
        super(CTAlexNet, self).__init__()
        self.conv1 = TaskConv2d(32, 3, 64, 4, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.conv2 = TaskConv2d(s, 64, 128, 3, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.conv3 = TaskConv2d(s, 128, 256, 3, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = TaskLinear(256 * self.smid * self.smid, 2048, bias=False)
        self.fc2 = TaskLinear(2048, 2048, bias=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, TaskLinear) or isinstance(module, TaskConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode='train'):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}

        for name, module in self.named_modules():
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, TaskLinear) or isinstance(module, TaskConv2d):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                # Only uncomment if bias is supported
                # if getattr(module, 'bias') is not None:
                #     task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                # else:
                #     task_mask[name + '.bias'] = None

        return task_mask

    def reparam_conditional_task(self):
        for name, module in self.named_modules():
            if 'last' in name:
                continue

            if isinstance(module, TaskLinear) or isinstance(module, TaskConv2d):
                # Reparameterize
                nn.init.xavier_normal_(module.tasker[0].weight)
                nn.init.xavier_normal_(module.tasker[3].weight)

                # Bias
                module.tasker[0].bias.data.fill_(0.01)
                module.tasker[3].bias.data.fill_(0.01)


class SubnetAlexNet(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet, self).__init__()
        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

            # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
        y = self.last[task_id](x)
        return y

    def init_masks(self):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
        return task_mask


class StructuredSubnetAlexNet(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(StructuredSubnetAlexNet, self).__init__()
        self.in_channel = []
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = StructuredSubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = StructuredSubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = StructuredSubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        self.fc2 = StructuredSubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

            # Constant none_masks
        self.none_masks = {}
        self.build_none_masks()

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        # x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        # import ipdb
        # ipdb.set_trace()

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

                    # We do not prune-out input layer!
            if name == 'conv1':
                continue

            if isinstance(module, StructuredSubnetLinear) or isinstance(module, StructuredSubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

    def build_none_masks(self):
        self.none_weight_masks, self.none_bias_masks = {}, {}
        for name, module in self.named_modules():
            if isinstance(module, StructuredSubnetConv2d) or isinstance(module, StructuredSubnetLinear):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None


class StructuredSubnetAlexNet_v2(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(StructuredSubnetAlexNet_v2, self).__init__()
        self.in_channel = []
        self.conv1 = StructuredSubnetConv2d_v2(3, 64, 4, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = StructuredSubnetConv2d_v2(64, 128, 3, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = StructuredSubnetConv2d_v2(128, 256, 2, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = StructuredSubnetLinear_v2(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        self.fc2 = StructuredSubnetLinear_v2(2048, 2048, sparsity=sparsity, bias=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

            # Constant none_masks
        self.none_masks = {}
        self.build_none_masks()

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, StructuredSubnetLinear_v2) or isinstance(module, StructuredSubnetConv2d_v2):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

    def build_none_masks(self):
        self.none_weight_masks, self.none_bias_masks = {}, {}
        for name, module in self.named_modules():
            if isinstance(module, StructuredSubnetLinear_v2) or isinstance(module, StructuredSubnetConv2d_v2):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None


class StructuredSubnetAlexNet_v3(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(StructuredSubnetAlexNet_v3, self).__init__()
        self.in_channel = []
        self.conv1 = StructuredSubnetConv2d_v2(3, 64, 4, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = StructuredSubnetConv2d_v2(64, 128, 3, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = StructuredSubnetConv2d_v2(128, 256, 2, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = StructuredSubnetLinear_v2(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        self.fc2 = StructuredSubnetLinear_v2(2048, 2048, sparsity=sparsity, bias=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

            # Constant none_masks
        self.none_masks = {}
        self.build_none_masks()

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, StructuredSubnetLinear_v2) or isinstance(module, StructuredSubnetConv2d_v2):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
        return task_mask

    def build_none_masks(self):
        self.none_weight_masks, self.none_bias_masks = {}, {}
        for name, module in self.named_modules():
            if isinstance(module, StructuredSubnetLinear_v2) or isinstance(module, StructuredSubnetConv2d_v2):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def reparam_scores(self):
        for name, module in self.named_modules():
            if 'last' in name:
                continue

            if isinstance(module, StructuredSubnetLinear_v2) or isinstance(module, StructuredSubnetConv2d_v2):
                # Reparameterize
                module.init_mask_parameters()

                ####################################################################################


################################# With Batchnorm ###################################
####################################################################################

class STLAlexNet_norm(nn.Module):
    def __init__(self, taskcla, task):
        super(STLAlexNet_norm, self).__init__()
        self.in_channel = []
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)

        self.fc3 = nn.Linear(2048, taskcla[task][1], bias=False)

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.fc2(x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))

        y = self.fc3(x)

        return y


class GPMAlexNet_norm(nn.Module):
    def __init__(self, taskcla):
        super(GPMAlexNet_norm, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048, n, bias=False))

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3'] = x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y

        # TODO: model


class SubnetAlexNet_norm(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_norm, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))
            self.last.append(nn.Linear(2048, n, bias=False))


        # Create new head model with 2 fc layers, take in input from fc2
        self.adapt_fc1 = SubnetLinear(2048*2, 1048, sparsity=1, bias=False)
        self.adapt_bn1 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)
        self.adapt_fc2 = SubnetLinear(1048, 1048, sparsity=1, bias=False)
        self.adapt_bn2 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)

        self.adapt_last = nn.Linear(1048, 100, bias=False)


        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == task_id or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x

            x_2 = self.adapt_fc1(output, weight_mask=None, bias_mask=None, mode='joint')
            x_2 = self.drop2(self.relu(self.adapt_bn1(x_2)))
            x_2 = self.adapt_fc2(x_2, weight_mask=None, bias_mask=None, mode='joint')
            x_2 = self.drop2(self.relu(self.adapt_bn2(x_2)))
            y = self.adapt_last(x_2)
            return y
        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn5(x)))
            y = self.last[task_id](x)
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

class SubnetAlexNet_extended(nn.Module):
    def __init__(self, taskcla, sparsity=0.5, layers=1):
        super(SubnetAlexNet_extended, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(SubnetAlexNet_NN_LEAF(layers=layers))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks
        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(self.bn5(x)))
        y = self.last[task_id](x)
        return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask




class SubnetAlexNet_NN_LEAF(nn.Module):
    def __init__(self, layers=1, input_size=2048, n_tasks=100, internal_size=512, sparsity=0.5):
        super(SubnetAlexNet_NN_LEAF, self).__init__()
        # Create new head model with 2 fc layers, take in input from fc2
        self.layers = layers
        self.relu = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(0.5)
        last_layer_input = input_size if self.layers == 0 else internal_size
        if layers > 0:
            self.adapt_fc1 = nn.Linear(input_size, internal_size, bias=False)
            self.adapt_bn1 = nn.BatchNorm1d(internal_size, track_running_stats=False, affine=False)
        if layers > 1:
            self.adapt_fc2 = nn.Linear(internal_size, internal_size, bias=False)
            self.adapt_bn2 = nn.BatchNorm1d(internal_size, track_running_stats=False, affine=False)
        if layers > 2:
            self.adapt_fc3 = nn.Linear(internal_size, internal_size, bias=False)
            self.adapt_bn3 = nn.BatchNorm1d(internal_size, track_running_stats=False, affine=False)

        self.adapt_last = nn.Linear(last_layer_input, n_tasks, bias=False)

    def forward(self, x_2, weight_mask={},mode="joint"):
        # Set for second mas
        if self.layers > 0:
            x_2 = self.adapt_fc1(x_2)
            x_2 = self.drop2(self.relu(self.adapt_bn1(x_2)))
        if self.layers > 1:
            x_2 = self.adapt_fc2(x_2)
            x_2 = self.drop2(self.relu(self.adapt_bn2(x_2)))
        if self.layers > 2:
            x_2 = self.adapt_fc3(x_2)
            x_2 = self.drop2(self.relu(self.adapt_bn3(x_2)))
        y = self.adapt_last(x_2)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask




class SubnetAlexNet_single(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_single, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))
            self.last.append(nn.Linear(2048, n, bias=False))


        # Create new head model with 2 fc layers, take in input from fc2
        self.adapt_fc1 = SubnetLinear(2048*2, 1048, sparsity=1, bias=False)
        self.adapt_bn1 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)
        # self.adapt_fc2 = SubnetLinear(1048, 1048, sparsity=1, bias=False)
        # self.adapt_bn2 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)

        self.adapt_last = nn.Linear(1048, 100, bias=False)


        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False, early_exit=False):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == task_id or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x

            x_2 = self.adapt_fc1(output, weight_mask=None, bias_mask=None, mode='joint')
            x_2 = self.drop2(self.relu(self.adapt_bn1(x_2)))
            # x_2 = self.adapt_fc2(x_2, weight_mask=None, bias_mask=None, mode='joint')
            # x_2 = self.drop2(self.relu(self.adapt_bn2(x_2)))
            y = self.adapt_last(x_2)
            return y
        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn5(x)))

            if early_exit:
                return x

            y = self.last[task_id](x)
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask


class SubnetAlexNet_triple(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_triple, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))
            self.last.append(nn.Linear(2048, n, bias=False))


        # Create new head model with 2 fc layers, take in input from fc2
        self.adapt_fc1 = SubnetLinear(2048*2, 1048, sparsity=1, bias=False)
        self.adapt_bn1 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)
        self.adapt_fc2 = SubnetLinear(1048, 1048, sparsity=1, bias=False)
        self.adapt_bn2 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)
        self.adapt_fc3 = SubnetLinear(1048, 1048, sparsity=1, bias=False)
        self.adapt_bn3 = nn.BatchNorm1d(1048, track_running_stats=False, affine=False)
        self.adapt_last = nn.Linear(1048, 100, bias=False)


        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == task_id or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x

            x_2 = self.adapt_fc1(output, weight_mask=None, bias_mask=None, mode='joint')
            x_2 = self.drop2(self.relu(self.adapt_bn1(x_2)))
            x_2 = self.adapt_fc2(x_2, weight_mask=None, bias_mask=None, mode='joint')
            x_2 = self.drop2(self.relu(self.adapt_bn2(x_2)))
            x_2 = self.adapt_fc3(x_2, weight_mask=None, bias_mask=None, mode='joint')
            x_2 = self.drop2(self.relu(self.adapt_bn3(x_2)))
            y = self.adapt_last(x_2)
            return y
        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn5(x)))
            y = self.last[task_id](x)
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

class SubnetAlexNet_NN(nn.Module):
    def __init__(self, taskcla, sparsity=0.5, overlap=True):
        super(SubnetAlexNet_NN, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))
            self.last.append(nn.Linear(2048, n, bias=False))


        # Create new head model with 2 fc layers, take in input from fc2
        # self.adapt_fc1 = SubnetLinear(2048, 2048, sparsity=1, bias=False)
        # self.adapt_bn1 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        # self.adapt_fc2 = SubnetLinear(2048, 2048, sparsity=1, bias=False)
        # self.adapt_bn2 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        #
        # self.adapt_last = nn.Linear(2048, 100, bias=False)


        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False, early_exit=False):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == 0 or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x

            y = self.last[task_id](output)
            return y

        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
            x = self.drop2(self.relu(self.bn5(x)))

            if task_id is None:
                return x, None

            y = self.last[task_id](x)

            if early_exit:
                return x, y
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask


class SubnetAlexNet_NN_overlap(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_NN_overlap, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False, consolidated_masks={}, weight_overlap=None):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == 0 or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x
            if False:
                new_layer = nn.Linear(2048, 100, bias=False)
                new_layer = new_layer.to('cuda')
                new_layer.weight.data.fill_(0)
                a = new_layer(output[:, 2048:])
                print(torch.sum(a > 0))
            # y = self.last[0](output[:, :2048])
            y = self.last[task_id](output)
            return y

        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv1.weight'), weight_overlap=weight_overlap)

            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv2.weight'), weight_overlap=weight_overlap)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv3.weight'), weight_overlap=weight_overlap)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode,
                         consolidated_masks=consolidated_masks.get('fc1.weight'), weight_overlap=weight_overlap)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode,
                         consolidated_masks=consolidated_masks.get('fc2.weight'), weight_overlap=weight_overlap)
            x = self.drop2(self.relu(self.bn5(x)))
            y = self.last[task_id](x)
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

class SubnetAlexNet_NN_overlap(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_NN_overlap, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False, consolidated_masks={}, weight_overlap=None, early_exit=False):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == 0 or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x
            if False:
                new_layer = nn.Linear(2048, 100, bias=False)
                new_layer = new_layer.to('cuda')
                new_layer.weight.data.fill_(0)
                a = new_layer(output[:, 2048:])
                print(torch.sum(a > 0))
            # y = self.last[0](output[:, :2048])
            y = self.last[task_id](output)
            return y

        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv1.weight'), weight_overlap=weight_overlap)

            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv2.weight'), weight_overlap=weight_overlap)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv3.weight'), weight_overlap=weight_overlap)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode,
                         consolidated_masks=consolidated_masks.get('fc1.weight'), weight_overlap=weight_overlap)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode,
                         consolidated_masks=consolidated_masks.get('fc2.weight'), weight_overlap=weight_overlap)
            x = self.drop2(self.relu(self.bn5(x)))
            if early_exit:
                return x
            y = self.last[task_id](x)
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

    def get_taylor(self):
        taylor_dict = {}
        for name, module in self.named_modules():
            if 'last' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                # Reparameterize
                taylor = module.get_taylor()
                # print(name, taylor)
                taylor_dict[name] = taylor
        return taylor_dict


class SubnetAlexNet_NN_long(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_NN_long, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False, consolidated_masks={}, weight_overlap=None):
        # Set for second mas
        org_x = x
        if mask is None:
            mask = self.none_masks

        if joint:
            output = None

            first_last_masks = {k: v for k, v in training_masks.items() if k == 0 or k == list(training_masks.keys())[-1]}
            for task_id_tmp, mask in first_last_masks.items():
                bsz = deepcopy(x.size(0))
                x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

                x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
                x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

                x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
                x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

                x = x.view(bsz, -1)
                x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn4(x)))

                x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
                x = self.drop2(self.relu(self.bn5(x)))
                output = torch.cat((output, x), dim=-1) if output is not None else x
                x = org_x
            if False:
                new_layer = nn.Linear(2048, 100, bias=False)
                new_layer = new_layer.to('cuda')
                new_layer.weight.data.fill_(0)
                a = new_layer(output[:, 2048:])
                print(torch.sum(a > 0))
            # y = self.last[0](output[:, :2048])
            y = self.last[task_id](output)
            return y

        else:
            bsz = deepcopy(x.size(0))
            x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv1.weight'), weight_overlap=weight_overlap)

            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv2.weight'), weight_overlap=weight_overlap)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode,
                           consolidated_masks=consolidated_masks.get('conv3.weight'), weight_overlap=weight_overlap)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode,
                         consolidated_masks=consolidated_masks.get('fc1.weight'), weight_overlap=weight_overlap)
            x = self.drop2(self.relu(self.bn4(x)))

            x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode,
                         consolidated_masks=consolidated_masks.get('fc2.weight'), weight_overlap=weight_overlap)
            x = self.drop2(self.relu(self.bn5(x)))
            y = self.last[task_id](x)
            return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

class SubnetAlexNet_NN_transfer(nn.Module):
    def __init__(self, model1, model2):
        super(SubnetAlexNet_NN_transfer, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.last = nn.ModuleList()
        self.last.append(nn.Linear(2048 *2, 100, bias=False))


    def forward(self, x, task_id, mask1=None, mask2=None, mode="train"):
        # Set for second mas
        org_x = x

        bsz = deepcopy(x.size(0))
        x = self.model1.conv1(x, weight_mask=mask1['conv1.weight'], bias_mask=mask1['conv1.bias'], mode=mode)
        x = self.model1.maxpool(self.model1.drop1(self.model1.relu(self.model1.bn1(x))))

        x = self.model1.conv2(x, weight_mask=mask1['conv2.weight'], bias_mask=mask1['conv2.bias'], mode=mode)
        x = self.model1.maxpool(self.model1.drop1(self.model1.relu(self.model1.bn2(x))))

        x = self.model1.conv3(x, weight_mask=mask1['conv3.weight'], bias_mask=mask1['conv3.bias'], mode=mode)
        x = self.model1.maxpool(self.model1.drop2(self.model1.relu(self.model1.bn3(x))))

        x = x.view(bsz, -1)
        x = self.model1.fc1(x, weight_mask=mask1['fc1.weight'], bias_mask=mask1['fc1.bias'], mode=mode)
        x = self.model1.drop2(self.model1.relu(self.model1.bn4(x)))

        x = self.model1.fc2(x, weight_mask=mask1['fc2.weight'], bias_mask=mask1['fc2.bias'], mode=mode)
        x = self.model1.drop2(self.model1.relu(self.model1.bn5(x)))

        # ----model 2---
        x_2 = self.model2.conv1(org_x, weight_mask=mask2['conv1.weight'], bias_mask=mask2['conv1.bias'], mode=mode)
        x_2 = self.model2.maxpool(self.model2.drop1(self.model2.relu(self.model2.bn1(x_2))))

        x_2 = self.model2.conv2(x_2, weight_mask=mask2['conv2.weight'], bias_mask=mask2['conv2.bias'], mode=mode)
        x_2 = self.model2.maxpool(self.model2.drop1(self.model2.relu(self.model2.bn2(x_2))))

        x_2 = self.model2.conv3(x_2, weight_mask=mask2['conv3.weight'], bias_mask=mask2['conv3.bias'], mode=mode)
        x_2 = self.model2.maxpool(self.model2.drop2(self.model2.relu(self.model2.bn3(x_2))))

        x_2 = x_2.view(bsz, -1)
        x_2 = self.model2.fc1(x_2, weight_mask=mask2['fc1.weight'], bias_mask=mask2['fc1.bias'], mode=mode)
        x_2 = self.model2.drop2(self.model2.relu(self.model2.bn4(x_2)))

        x_2 = self.model2.fc2(x_2, weight_mask=mask2['fc2.weight'], bias_mask=mask2['fc2.bias'], mode=mode)
        x_2 = self.model2.drop2(self.model2.relu(self.model2.bn5(x_2)))

        output = torch.cat((x, x_2), dim=1)

        y = self.last[0](output)
        return y




# AB_acc_list [53.8, 54.2, 54.4, 54.6, 54.2, 54.4, 54.2, 54.2, 54.4, 54.8, 54.6, 55.0, 54.2, 54.6, 54.4, 53.8, 54.8, 53.8, 54.8]

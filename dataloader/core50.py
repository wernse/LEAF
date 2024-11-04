import os
import pickle as pkl
import logging
from hashlib import md5
import numpy as np
import torch
from PIL import Image
import time

from torchvision.transforms import transforms

def shuffle_data(x, y, seed):
    perm_inds = np.arange(0, x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y

def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))
    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]



core50_ntask = {
    'ni': 8,
    'nc': 9,
    'nic': 79,
    'nicv2_79': 79,
    'nicv2_196': 196,
    'nicv2_391': 391
}


class CORE50:
    def __init__(self, args):
        self.args = args
        self.root = './data/core50'
        self.scenario = 'nc'
        self.download_load()


    def download_load(self):
        print("Loading paths...")
        with open(os.path.join(self.root, 'paths.pkl'), 'rb') as f:
            self.paths = pkl.load(f)

        print("Loading LUP...")
        with open(os.path.join(self.root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        print("Loading labels...")
        with open(os.path.join(self.root, 'labels.pkl'), 'rb') as f:
            self.labels = pkl.load(f)


    def new_task(self):

        data = {}
        cur_run = -1
        self.test_set = []
        print('Loading test set...')
        test_idx_list = self.LUP[self.scenario][cur_run][-1]

        # test paths
        test_paths = []
        for idx in test_idx_list:
            test_paths.append(os.path.join(self.root, self.paths[idx]))

        # test imgs
        self.test_data = self.get_batch_from_paths(test_paths)
        self.test_label = np.asarray(self.labels[self.scenario][cur_run][-1])

        self.task_labels = self.labels[self.scenario][cur_run][:-1]
        for labels in self.task_labels:
            labels = list(set(labels))
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            self.test_set.append((x_test, y_test))

        n_tasks = len(self.test_set)
        for cur_task in range(n_tasks):
            print(cur_task)
            data[cur_task] = {}
            s = time.time()
            train_idx_list = self.LUP[self.scenario][cur_run][cur_task]
            print("Loading data...")
            # Getting the actual paths
            train_paths = []
            for idx in train_idx_list:
                train_paths.append(os.path.join(self.root, self.paths[idx]))
            # loading imgs
            train_x = self.get_batch_from_paths(train_paths)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_y = self.labels[self.scenario][cur_run][cur_task]
            train_y = np.asarray(train_y)
            # get val set
            train_x_rdm, train_y_rdm = shuffle_data(train_x, train_y, self.args.seed)
            val_size = int(len(train_x_rdm) * self.args.pc_valid)
            val_data_rdm, val_label_rdm = train_x_rdm[:val_size], train_y_rdm[:val_size]
            train_data_rdm, train_label_rdm = train_x_rdm[val_size:], train_y_rdm[val_size:]

            train_data_rdm = torch.stack([transform(img) for img in train_data_rdm])
            val_data_rdm = torch.stack([transform(img) for img in val_data_rdm])
            test_data_rdm = torch.stack([transform(img) for img in self.test_set[cur_task][0]])

            e = time.time()
            print('loading time {}'.format(str(e-s)))
            data[cur_task]['train'] = {'x': train_data_rdm, 'y': torch.tensor(train_label_rdm)}
            data[cur_task]['test'] = {'x': test_data_rdm, 'y': torch.tensor(self.test_set[cur_task][1])}
            data[cur_task]['valid'] = {'x': val_data_rdm, 'y': torch.tensor(val_label_rdm)}
            data[cur_task]['n_tasks'] = len(set(train_label_rdm))
            print(train_data_rdm.shape, test_data_rdm.shape, val_data_rdm.shape)
        output_info = []
        n = 0
        for t in data.keys():
            size = data[t]['n_tasks']
            output_info.append((t, size))
            n += size
        data['ncla'] = n
        input_size = [128, 128, 3]
        return data, output_info, input_size, n_tasks, n

    """
        def new_task(self):

            data = {}
            n_tasks = 10
            cur_run = 0
            for cur_task in range(n_tasks):
                print(cur_task)
                data[cur_task] = {}
                s = time.time()
                train_idx_list = self.LUP[self.scenario][cur_run][cur_task]
                print("Loading data...")
                # Getting the actual paths
                train_paths = []
                for idx in train_idx_list:
                    train_paths.append(os.path.join(self.root, self.paths[idx]))
                # loading imgs
                train_x = self.get_batch_from_paths(train_paths)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                train_y = self.labels[self.scenario][cur_run][cur_task]
                train_y = np.asarray(train_y)
                # get val set
                train_x_rdm, train_y_rdm = shuffle_data(train_x, train_y, self.args.seed)
                val_size = int(len(train_x_rdm) * self.args.pc_valid)
                test_size = int(len(train_x_rdm) * 0.2)

                val_data_rdm, val_label_rdm = train_x_rdm[:val_size], train_y_rdm[:val_size]
                train_data_rdm, train_label_rdm = train_x_rdm[val_size:], train_y_rdm[val_size:]

                test_data_rdm, test_label_rdm = train_data_rdm[:test_size], train_label_rdm[:test_size]
                train_data_rdm, train_label_rdm = train_data_rdm[test_size:], train_label_rdm[test_size:]

                train_data_rdm = torch.stack([transform(img) for img in train_data_rdm])
                val_data_rdm = torch.stack([transform(img) for img in val_data_rdm])
                test_data_rdm = torch.stack([transform(img) for img in test_data_rdm])

                e = time.time()
                print('loading time {}'.format(str(e-s)))
                data[cur_task]['train'] = {'x': train_data_rdm, 'y': torch.tensor(train_label_rdm)}
                data[cur_task]['test'] = {'x': test_data_rdm, 'y': torch.tensor(test_label_rdm)}
                data[cur_task]['valid'] = {'x': val_data_rdm, 'y': torch.tensor(val_label_rdm)}
                data[cur_task]['n_tasks'] = len(set(train_label_rdm))
                print(len(train_label_rdm), len(test_label_rdm), len(val_label_rdm))"""

    def new_run(self, **kwargs):
        cur_run = kwargs['cur_run']
        self.setup(cur_run)


    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False):
        """ Given a number of abs. paths it returns the numpy array
        of all the images. """

        # Getting root logger
        log = logging.getLogger('mylogger')

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, 'rb') as f:
                    npzfile = np.load(f)
                    x, y = npzfile['x']
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, 'rb') as f:
                    x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')
                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, 'wb') as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x



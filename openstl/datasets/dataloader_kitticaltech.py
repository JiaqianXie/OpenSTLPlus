import os
import os.path as osp
import cv2
import random
import numpy as np
from skimage.transform import resize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader
import shutil

try:
    import hickle as hkl
except ImportError:
    hkl = None

import matplotlib.pyplot as plt

class KittiCaltechDataset(Dataset):
    """KittiCaltech <https://dl.acm.org/doi/10.1177/0278364913491297>`_ Dataset"""

    def __init__(self, datas, indices, pre_seq_length, aft_seq_length,
                 require_back=False, use_augment=False, data_name='kitticaltech', augment_params=None,
                 visualize_data=False, vis_dir=None):
        super(KittiCaltechDataset, self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1, 2)
        self.visualize_data = visualize_data
        self.vis_dir = vis_dir
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.require_back = require_back
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name
        self.augment_params = augment_params

        if visualize_data:
            if os.path.exists(vis_dir):
                shutil.rmtree(vis_dir)
            os.makedirs(vis_dir)
            os.makedirs(os.path.join(vis_dir, 'input'), exist_ok=True)
            os.makedirs(os.path.join(vis_dir, 'gt'), exist_ok=True)

    def _augment_seq(self, imgs, crop_scale=0.95, augment_params=None):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [12, 3, 128, 160]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        if "use_crop" in augment_params and augment_params["use_crop"]:
            x = np.random.randint(0, ih - h + 1)
            y = np.random.randint(0, iw - w + 1)
            imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if "use_flip" in augment_params and augment_params["use_flip"]:
            if random.randint(0, 1):
                imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        # Random Masking
        if "use_mask" in augment_params and augment_params["use_mask"]:
            max_mask_ratio = augment_params.get('max_mask_ratio', 0.1)
            mask_prob = augment_params.get('mask_prob', 0.5)
            max_num_masks = augment_params.get('max_num_masks', 3)
            if random.random() < mask_prob:
                # Decide the number of masks to apply
                num_masks = random.randint(1, max_num_masks)
                for _ in range(num_masks):
                    # Define mask size
                    mask_h = np.random.randint(int(h * max_mask_ratio * 0.5), int(h * max_mask_ratio))
                    mask_w = np.random.randint(int(w * max_mask_ratio * 0.5), int(w * max_mask_ratio))
                    # Random position for the mask
                    top = np.random.randint(0, h - mask_h)
                    left = np.random.randint(0, w - mask_w)
                    # Apply the mask (set pixels to zero)
                    imgs[:, :, top:top + mask_h, left:left + mask_w] = 0

        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = end1 + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1, ::]).float()
        labels = torch.tensor(self.datas[end1:end2, ::]).float()
        if self.use_augment:
            imgs = torch.cat([data, labels], dim=0)
            data = imgs[:self.pre_seq_length, ...]
            data = self._augment_seq(data, crop_scale=0.94, augment_params=self.augment_params)
            labels = imgs[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length, ...]
            if self.visualize_data:
                self.visualize(data, i, "input")
                self.visualize(labels, i, "gt")
        return data, labels

    def visualize(self, data, i, folder):
        """
        Visualizes the input image (i.e., the variable "data" in __getitem__ function).

        Args:
            i (int): Index of the data to visualize.
        """
        # Assuming data shape is [T, C, H, W], where T is the time dimension
        for t in range(data.shape[0]):
            img = data[t].permute(1, 2, 0).numpy() * 255.0  # Convert from [C, H, W] to [H, W, C]
            plt.imshow(img.astype(np.uint8))
            plt.title(f"Frame {t + 1}")
            plt.axis('off')
            save_path = os.path.join(self.vis_dir, folder, f"video{i}_frame_t{t + 1}.png")
            plt.savefig(save_path)
            plt.close()

def process_im(im, desired_sz):
    # cite the `process_im` code from PredNet, Thanks!
    # https://github.com/coxlab/prednet/blob/master/process_kitti.py
    target_ds = float(desired_sz[0]) / im.shape[0]
    im = resize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), preserve_range=True)
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


class DataProcess(object):

    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.seq_len = input_param['seq_length']
        self.input_shape = input_param['input_shape']  # (128, 160)

    def load_data(self, mode='train'):
        """Loads the dataset.
        Args:
          paths: paths of train/test dataset.
          mode: Training or testing.
        Returns:
          A dataset and indices of the sequence.
        """
        if mode == 'train' or mode == 'val':
            kitti_root = self.paths['kitti']
            data = hkl.load(osp.join(kitti_root, 'X_' + mode + '.hkl'))
            data = data.astype('float') / 255.0
            fileidx = hkl.load(
                osp.join(kitti_root, 'sources_' + mode + '.hkl'))

            indices = []
            index = len(fileidx) - 1
            while index >= self.seq_len - 1:
                if fileidx[index] == fileidx[index - self.seq_len + 1]:
                    indices.append(index - self.seq_len + 1)
                    index -= self.seq_len - 1
                index -= 1

        elif mode == 'test':
            caltech_root = self.paths['caltech']
            # find the cache file
            caltech_cache = osp.join(caltech_root, 'data_cache.npy')
            if osp.exists(caltech_cache):
                data = np.load(caltech_cache).astype('float') / 255.0
                indices = np.load(osp.join(caltech_root, 'indices_cache.npy'))
            else:
                print(f'loading KittiCaltech from {caltech_root}, which requires some times...')
                data = []
                fileidx = []
                for seq_id in os.listdir(caltech_root):
                    if osp.isdir(osp.join(caltech_root, seq_id)) is False:
                        continue
                    for item in os.listdir(osp.join(caltech_root, seq_id)):
                        seq_file = osp.join(caltech_root, seq_id, item)
                        print(seq_file)
                        cap = cv2.VideoCapture(seq_file)
                        cnt_frames = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            cnt_frames += 1
                            if cnt_frames % 3 == 0:
                                frame = process_im(frame, self.input_shape) / 255.0
                                data.append(frame)
                                fileidx.append(seq_id + item)
                data = np.asarray(data)

                indices = []
                index = len(fileidx) - 1
                while index >= self.seq_len - 1:
                    if fileidx[index] == fileidx[index - self.seq_len + 1]:
                        indices.append(index - self.seq_len + 1)
                        index -= self.seq_len - 1
                    index -= 1

                # save the cache file
                data_cache = data * 255
                np.save(caltech_cache, data_cache.astype('uint8'))
                indices_cache = np.asarray(indices)
                np.save(osp.join(caltech_root, 'indices_cache.npy'), indices_cache.astype('int32'))

        return data, indices


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=1, in_shape=[10, 3, 128, 160],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False, augment_params=None,
              visualize_data=False, vis_dir=None):
    print(vis_dir)
    if os.path.exists(osp.join(data_root, 'kitti_hkl')):
        input_param = {
            'paths': {'kitti': osp.join(data_root, 'kitti_hkl'),
                    'caltech': osp.join(data_root, 'caltech')},
            'seq_length': (pre_seq_length + aft_seq_length),
            'input_data_type': 'float32',
            'input_shape': (in_shape[-2], in_shape[-1]) if in_shape is not None else (128, 160),
        }
        input_handle = DataProcess(input_param)
        train_data, train_idx = input_handle.load_data('train')
        test_data, test_idx = input_handle.load_data('test')
    elif os.path.exists(osp.join(data_root, 'kitticaltech_npy')):
        train_data = np.load(osp.join(data_root, 'kitticaltech_npy', 'train_data.npy'))
        train_idx = np.load(osp.join(data_root, 'kitticaltech_npy', 'train_idx.npy'))
        test_data = np.load(osp.join(data_root, 'kitticaltech_npy', 'test_data.npy'))
        test_idx = np.load(osp.join(data_root, 'kitticaltech_npy', 'test_idx.npy'))
    else:
        assert False and "Invalid data_root for kitticaltech dataset"

    train_set = KittiCaltechDataset(
        train_data, train_idx, pre_seq_length, aft_seq_length, use_augment=use_augment, augment_params=augment_params,
        visualize_data=visualize_data,vis_dir=vis_dir)
    test_set = KittiCaltechDataset(
        test_data, test_idx, pre_seq_length, aft_seq_length, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = None
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                val_batch_size=16,
                data_root='../../data/',
                num_workers=4,
                pre_seq_length=12, aft_seq_length=1)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break

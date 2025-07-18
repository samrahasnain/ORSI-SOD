import os
import cv2
import torch
from torch.utils import data
import numpy as np
import random

random.seed(10)

class ImageDataTrain(data.Dataset):
    def __init__(self, image_dir, label_dir, image_size, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform

        self.image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.label_list = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.png'))])

        assert len(self.image_list) == len(self.label_list), "Mismatch between number of images and labels"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        label_path = os.path.join(self.label_dir, self.label_list[index])

        sal_image, im_size = load_image(image_path, self.image_size)
        sal_label, sal_edge = load_sal_label(label_path, self.image_size)

        sal_image, sal_image, sal_label = cv_random_crop(sal_image, sal_image, sal_label, self.image_size)

        sal_image = sal_image.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))
        sal_edge = sal_edge.transpose((2, 0, 1))

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)

        sample = {'sal_image': sal_image, 'sal_depth': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge}
        return sample


class ImageDataTest(data.Dataset):
    def __init__(self, image_dir, label_dir, image_size):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size

        self.image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.label_list = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.png'))])

        assert len(self.image_list) == len(self.label_list), "Mismatch between number of test images and labels"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        label_path = os.path.join(self.label_dir, self.label_list[index])

        image, im_size = load_image_test(image_path, self.image_size)
        label, _ = load_image_test(label_path, self.image_size)

        image = torch.Tensor(image)
        label = torch.Tensor(label)

        return {
            'image': image,
            'label': label,
            'name': self.image_list[index],
            'size': im_size
        }


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path, image_size):
    if not os.path.exists(path):
        print(f'File {path} does not exist')
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    return in_, im_size


def load_image_test(path, image_size):
    if not os.path.exists(path):
        print(f'File {path} does not exist')
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path, image_size):
    if not os.path.exists(path):
        print(f'File {path} does not exist')
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    gX = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    combined = np.array(combined, dtype=np.float32)
    combined = cv2.resize(combined, (image_size, image_size))
    combined = combined / 255.0
    combined = combined[..., np.newaxis]

    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label, (image_size, image_size))
    label = label / 255.0
    label = label[..., np.newaxis]

    return label, combined


def cv_random_crop(image, depth, label, image_size):
    crop_size = int(0.0625 * image_size)
    croped = image_size - crop_size
    top = random.randint(0, crop_size)
    left = random.randint(0, crop_size)

    image = image[top: top + croped, left: left + croped, :]
    depth = depth[top: top + croped, left: left + croped, :]
    label = label[top: top + croped, left: left + croped, :]
    image = cv2.resize(image, (image_size, image_size))
    depth = cv2.resize(depth, (image_size, image_size))
    label = cv2.resize(label, (image_size, image_size))
    label = label[..., np.newaxis]

    return image, depth, label


def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ = in_ / 255.0
    in_ -= np.array((0.485, 0.456, 0.406))
    in_ /= np.array((0.229, 0.224, 0.225))
    return in_

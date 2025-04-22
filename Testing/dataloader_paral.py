import os
import torch
import numpy as np
import imageio
import cv2
from torch.utils.data import Dataset


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class cityscapesLoader(Dataset):
    colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
        [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    label_colours = dict(zip(range(19), colors))

    def __init__(self, img_path, in_size=(769, 1537)):
        self.img_path = img_path
        self.files = sorted(recursive_glob(rootdir=self.img_path, suffix=".png"))
        self.target_size = (in_size[1], in_size[0])  # width x height
        self.mean = np.array([.485, .456, .406])
        self.std = np.array([.229, .224, .225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img_name = os.path.basename(img_path)
        folder = os.path.basename(os.path.dirname(img_path))

        img = imageio.imread(img_path)
        ori_h, ori_w = img.shape[:2]
        target_w, target_h = self.target_size

        # Compute scale factor (fit inside target size while preserving aspect)
        scale = min(target_w / ori_w, target_h / ori_h)
        new_w, new_h = int(ori_w * scale), int(ori_h * scale)

        # Resize while maintaining aspect ratio
        img_resized = cv2.resize(img, (new_w, new_h))

        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Pad image to match target size
        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # Normalize
        img_normalized = img_padded / 255.0
        img_normalized = (img_normalized - self.mean) / self.std
        img_transposed = img_normalized.transpose(2, 0, 1)  # HWC → CHW
        img_tensor = torch.from_numpy(img_transposed).float()

        padding = (pad_top, pad_bottom, pad_left, pad_right)
        resized_shape = (new_h, new_w)
        ori_size = (ori_h, ori_w)

        return img_tensor.unsqueeze(0), img_name, folder, ori_size, padding, resized_shape

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, len(self.colors)):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

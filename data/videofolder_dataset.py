# ====================================================================
# ===================== identical with MoCoGAN =======================
# ====================================================================

import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import PIL


class VideoFolderDataset(torch.utils.data.Dataset):
    """
    from https://github.com/sergeytulyakov/mocogan
    """
    def __init__(self, folder, cache, min_len=32):
        print(self.name())
        dataset = ImageFolder(folder)
        self.total_frames = 0
        self.lengths = []
        self.images = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'r') as f:
                self.images, self.lengths = pickle.load(f)
        else:
            for idx, (im, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                shorter, longer = min(im.width, im.height), max(im.width, im.height)
                length = longer // shorter
                if length >= min_len:
                    self.images.append((img_path, categ))
                    self.lengths.append(length)

            if cache is not None:
                with open(cache, 'w') as f:
                    pickle.dump((self.images, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def __getitem__(self, item):
        path, label = self.images[item]
        im = PIL.Image.open(path)
        return im, label, None

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'VideoFolderDataset'
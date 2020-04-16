# ====================================================================
# ===================== identical with MoCoGAN =======================
# ====================================================================

import numpy as np
import torch.utils.data


class ImageDataset(torch.utils.data.Dataset):
    """
        from https://github.com/sergeytulyakov/mocogan
    """
    def __init__(self, dataset, transform=None):
        print(self.name())
        self.dataset = dataset

        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, action, actor, name = self.dataset[video_id]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]

        if horizontal:
            i_from, i_to = video.shape[0] * frame_num, video.shape[0] * (frame_num + 1)
            frame = video[:, i_from: i_to, ::]
        else:
            i_from, i_to = video.shape[1] * frame_num, video.shape[1] * (frame_num + 1)
            frame = video[i_from: i_to, :, ::]

        if frame.shape[0] == 0:
            print("video {}. From {} to {}. num {}".format(video.shape, i_from, i_to, item))

        return {"images": self.transforms(frame), "categories": action, 'actors': actor, 'names': name}

    def __len__(self):
        return self.dataset.cumsum[-1]

    def name(self):
        return 'ImageDataset'

import os
import numpy as np
import torch.utils.data


class GeneratedDataset(torch.utils.data.Dataset):
    """
    Dataloader for the Generated Videos
    """
    def __init__(self, dataroot, is_actor, actor_set, action_set):
        """
        Initialization method
        :param dataroot: str, the path for the generated data
        :param is_actor: bool, true if the video is generated provided the first frame
        :param actor_set: list, the actor set
        :param action_set: list, the action set
        """
        print(self.name())
        self.is_actor = is_actor
        self.dataroot = dataroot
        self.cases = []
        self.action_set = action_set
        self.actor_set = actor_set

        # get the list of generated videos
        for video in os.listdir(dataroot):
            if is_actor:
                # e.g. shahar_side_001.npy
                criterion = video.endswith('npy') and (not video.startswith('none'))
            else:
                # e.g. none_side_001.npy
                criterion = video.endswith('npy') and video.startswith('none')
            if criterion:
                self.cases.append(video)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, item):
        """
        Get the item
        :param item: int, the query index
        :return video: ndarray with size [h, w*(end-start+1), c]
                action: int, the action index
                actor: int, the actor index
                name: str, the file name without extension
        """
        # load the video npy file
        video_name = self.cases[item]
        npy_path = os.path.join(self.dataroot, video_name)
        video = torch.from_numpy(np.load(npy_path))
        video = video.permute(1, 0, 2, 3)

        # get action and actor label
        tokens = video_name.split('_')
        action_name = tokens[1]
        if action_name not in self.action_set:
            self.action_set.append(action_name)
        action = self.action_set.index(action_name)
        if self.is_actor:
            actor_name = tokens[0]
            if actor_name not in self.actor_set:
                self.actor_set.append(actor_name)
            actor = self.actor_set.index(actor_name)
        else:
            actor = -1
        return {"images": video, "categories": action, 'actors': actor, 'names': video_name.split('.')[0]}

    def name(self):
        return 'GeneratedDataset'
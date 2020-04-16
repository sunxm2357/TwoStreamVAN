import numpy as np
import torch.utils.data
import pdb


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        """
        Initialization
        :param dataset: dataloader object, the dataloader for the dataset
        :param video_length: int, the length of the generated video
        :param every_nth: int, the frequency to read the frames
        :param transform: a list of functions, define the image transform
        """
        print(self.name())
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        """
        The loading method when querying the index
        :param item: the query index
        :return: 'images': a tensor of [c, video_length, h, w],
                 'categories': int, action class label,
                 'actors' : int, actor label,
                 'names': str, the file name
        """

        # get the video, action, actor and name from the specific dataloader
        video, action, actor, name = self.dataset[item]
        video = np.array(video)

        # get the video length and the concatenating direction
        horizontal = video.shape[1] > video.shape[0]
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        video_len = longer // shorter

        # prepare the every_nth according to the dataset and the action category
        if self.dataset.name == 'WeizmannDataset':
            every_nth = 1 if action in [6, 7, 8] else self.every_nth
        elif self.dataset.name == 'SynActionDataset':
            #                 0     1        2         3        4           5       6       7       8       9
            #  action_set = ['bow', 'cheer', 'clap', 'climb', 'crossstep', 'hit', 'hook', 'jump', 'kick', 'knee',
            #               'pick', 'pitch', 'push', 'roll', 'run', 'squat', 'stall', 'standup', 'walk', 'wave']
            if action in [0, 6, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]:
                every_nth = 4
            elif action in [5, 7, 8, 11]:
                every_nth = 6
            else:
                every_nth = 2
        else:
            every_nth = self.every_nth

        # videos can be of various length, we randomly sample sub-sequences
        if video_len > self.video_length * every_nth:
            needed = every_nth * (self.video_length - 1)
            gap = video_len - needed
            if self.dataset.mode == 'Test':
                start = 0
            else:
                start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            print(longer, shorter)
            raise Exception("Length is too short id - {}, len - {}".format(item, video_len))

        frames = np.split(video, video_len, axis=1 if horizontal else 0)
        selected = np.array([frames[s_id] for s_id in subsequence_idx])
        return {"images": self.transforms(selected), "categories": action, 'actors': actor, 'names': name}

    def __len__(self):
        return len(self.dataset)

    def name(self):
        return "VideoDataset"

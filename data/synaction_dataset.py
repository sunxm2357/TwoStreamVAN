import sys
sys.path.insert(0, '..')
import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
import cv2
from utils.util import makedir
import glob
import imageio
import pdb


class SynActionDataset(torch.utils.data.Dataset):
    """
    Dataloader for the SynAction Dataset
    """
    def __init__(self, dataroot, textroot, video_len, image_size, crop, mode="Train"):
        """
        Initialization and get the data list
        :param dataroot: str, the path for the stored data
        :param textroot: str, the path to write the data list
        :param video_len: int, the length of the generated video
        :param image_size: int, the spatial size of the image
        :param crop: bool, true if random cropping
        :param mode: ['Train', 'Test']
        """
        print(self.name())
        self.dataroot = dataroot
        self.textroot = textroot
        self.video_len = video_len
        self.crop = crop
        self.mode = mode
        self.image_size = image_size
        self.load_cache()

    def load_cache(self):
        """
        load the cache file if exists or create the cache file
        """
        cache = os.path.join(self.dataroot, 'cache_%s_%d.db' % (self.mode, self.video_len))
        if cache is not None and os.path.exists(cache):
            # load the cache
            with open(cache, 'rb') as f:
                self.lines, self.lengths, self.actor_set, self.action_set = pickle.load(f)
        else:
            # build the cache file
            self.lines, self.lengths, self.actor_set, self.action_set = self.build_dataset()
            makedir(self.textroot)
            text_file = os.path.join(self.textroot, '%s_list_%d.txt' % (self.mode, self.video_len))
            with open(text_file, 'w+') as f:
                f.writelines(self.lines)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.lines, self.lengths, self.actor_set, self.action_set), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def build_dataset(self):
        """
        Get the {name, start_idx, end_idx} and the length of each video;
        Get the action set and actor set of each dataset
        :return: lists of {name, start_idx, end_idx}; lengths of videos; action set and actor set
        """
        lines = []
        lengths = []
        actor_set = ['douglas', 'kachujin', 'liam', 'lola', 'malcolm', 'regina', 'remy', 'shae', 'stefani', 'swat']
        action_set = ['bow', 'cheer', 'clap', 'climb', 'crossstep', 'hit', 'hook', 'jump', 'kick', 'knee',
                      'pick', 'pitch', 'push', 'roll', 'run', 'squat', 'stall', 'standup', 'walk', 'wave']

        for video in tqdm.tqdm(glob.glob(os.path.join(self.dataroot, '*.npy'))):
            video_name = video.split('/')[-1]
            # <actor>_<action>_<seq_id>_<split_id>.npy
            # build the action and actor sets
            video_prefix = video_name.split('.')[0]
            tokens = video_prefix.split('_')
            action = action_set.index(tokens[1])
            actor = actor_set.index(tokens[0])
            split_id = int(tokens[-1])

            # get the sampling frequency based on the action class
            if action in [0, 6, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]:
                every_nth = 4
            elif action in [5, 7, 8, 11]:
                every_nth = 6
            else:
                every_nth = 2

            unique_seq_id = (actor * len(action_set) + action) * 2 + int(tokens[2])
            if self.mode == 'Test':
                # get line and length for the test mode
                if unique_seq_id % 15 == split_id:
                    npy_file = np.load(video)
                    total_len = len(npy_file)
                    if total_len > (self.video_len - 1) * every_nth + 1:
                        lines.append('%s %d %d \n' % (video_prefix, 0, total_len-1))
                        lengths.append(total_len)
            else:
                # get line and length for the training mode
                if unique_seq_id % 15 != split_id:
                    npy_file = np.load(video)
                    total_len = len(npy_file)
                    if total_len > (self.video_len - 1) * every_nth + 1:
                        lines.append('%s %d %d \n' % (video_prefix, 0, total_len-1))
                        lengths.append(total_len)

        return lines, lengths, actor_set, action_set

    def get_sequence(self, name, start_idx, end_idx):
        """
        Construct the sequence from the origin video
        :param name: str, the video name
        :param start_idx: int, the start index
        :param end_idx: int, the end index
        :return: sequence, ndarray, with size [h, w, c, end-start+1]
        """
        picked = range(start_idx, end_idx + 1)
        name = name + '.npy'
        video = np.load(os.path.join(self.dataroot, name))
        sequence = video[picked, :, :, :]
        return sequence

    def preprocess(self, sequence):
        """
        crop
        :param sequence: ndarray, with size [h, w, c, end_idx - start_idx +1]
        :return output: ndarray, with size [h, w*(end-start+1), c]
        """
        video_len, height, width, channel = sequence.shape
        # crop
        if self.crop:
            tl_h = np.random.randint(low=0, high=7)
            tl_w = np.random.randint(low=0, high=7)
        else:
            tl_h = 4
            tl_w = 4
        cropped_seq = sequence[:, tl_h: tl_h + self.image_size, tl_w: tl_w + self.image_size]
        output = cropped_seq.reshape(-1, self.image_size, channel)
        return output

    def __getitem__(self, item):
        """
        Get the item
        :param item: int, the query index
        :return video: ndarray with size [h, w*(end-start+1), c]
                action: int, the action index
                actor: int, the actor index
                name: str, the file name without extension
        """
        line = self.lines[item]
        name, start_idx, end_idx = line.split()
        sequence = self.get_sequence(name, int(start_idx), int(end_idx))
        video = self.preprocess(sequence)
        tokens = name.split('.')[0].split('_')
        action = self.action_set.index(tokens[1])
        actor = self.actor_set.index(tokens[0])
        return video, action, actor, name.split('.')[0]

    def __len__(self):
        return len(self.lines)

    def name(self):
        return 'SynActionDataset'
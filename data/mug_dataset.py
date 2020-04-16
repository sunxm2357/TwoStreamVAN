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
import pdb


class MUGDataset(torch.utils.data.Dataset):
    """
    Dataloader for the MUG Dataset
    """
    def __init__(self, dataroot, textroot, video_len, image_size, every_nth, crop, mode="Train"):
        """
        Initialization and get the data list
        :param dataroot: str, the path for the stored data
        :param textroot: str, the path to write the data list
        :param video_len: int, the length of the generated video
        :param image_size: int, the spatial size of the image
        :param every_nth: int, the frequency to sample frames
        :param crop: bool, true if random cropping
        :param mode: ['Train', 'Test']
        """
        print(self.name())
        self.dataroot = dataroot
        self.textroot = textroot
        self.video_len = video_len
        self.every_nth = every_nth
        self.crop = crop
        self.mode = mode
        self.image_size = image_size
        self.load_cache()

    def load_cache(self):
        """
        load the cache file if exists or create the cache file
        """
        cache = os.path.join(self.dataroot, 'cache_%s_%d.db' % (self.mode, self.video_len * self.every_nth))
        if cache is not None and os.path.exists(cache):
            # load the cache
            with open(cache, 'rb') as f:
                self.lines, self.lengths, self.actor_set, self.action_set = pickle.load(f)
        else:
            # build the cache file
            self.lines, self.lengths, self.actor_set, self.action_set = self.build_dataset()
            makedir(self.textroot)
            text_file = os.path.join(self.textroot, '%s_list_%d.txt' % (self.mode, self.video_len * self.every_nth))
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
        actor_set =[]
        action_set = []

        for video in tqdm.tqdm(glob.glob(os.path.join(self.dataroot, '*.npy'))):
            video_name = video.split('/')[-1]
            # <actor>_<action>_<seq_id>_<split_id>.npy
            # build the action and actor sets
            video_prefix = video_name.split('.')[0]
            tokens = video_prefix.split('_')
            if tokens[0] not in actor_set:
                actor_set.append(tokens[0])
            if tokens[1] not in action_set:
                action_set.append(tokens[1])
            split_id = int(tokens[-1])
            if self.mode == 'Test':
                # get line and length for the test mode
                if split_id == 3:
                    npy_file = np.load(video)
                    total_len = len(npy_file)
                    if total_len > (self.video_len - 1) * self.every_nth + 1:
                        lines.append('%s %d %d \n' % (video_prefix, 0, total_len-1))
                        lengths.append(total_len)
            else:
                # get line and length for the training mode
                if split_id != 3:
                    npy_file = np.load(video)
                    total_len = len(npy_file)
                    if total_len > (self.video_len - 1) * self.every_nth + 1:
                        lines.append('%s %d %d \n' % (video_prefix, 0, total_len-1))
                        lengths.append(total_len)

        return lines, lengths, actor_set, action_set

    def get_sequence(self, video, start_idx, end_idx):
        """
        Construct the sequence from the origin video
        :param video: ndarray, with size [h, w, c, t]
        :param start_idx: int, the start index
        :param end_idx: int, the end index
        :return: sequence, ndarray, with size [h, w, c, end-start+1]
        """
        picked = range(start_idx, end_idx + 1)
        sequence = video[picked, :, :, :]
        return sequence

    def preprocess(self, sequence):
        """
        crop and resize
        :param sequence: ndarray, with size [h, w, c, end_idx - start_idx +1]
        :return output: ndarray, with size [h, w*(end-start+1), c]
        """
        video_len, height, width, _,  = sequence.shape
        h, w = int(height * 0.9), int(width * 0.9)
        # crop
        if self.crop:
            tl_h = np.random.randint(low=0, high=height - h)
            tl_w = np.random.randint(low=0, high=width - w)
        else:
            tl_h = int((height - h) // 2)
            tl_w = int((width - w) // 2)
        cropped_seq = sequence[:, tl_h: tl_h + h, tl_w: tl_w + w]

        # resize
        frames = np.split(cropped_seq, video_len, axis=0)
        resized_frames = []
        for frame in frames:
            resized_frames.append(cv2.resize(frame.squeeze(), (self.image_size, self.image_size)))
        output = np.concatenate(resized_frames, axis=1)
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
        name = name + '.npy'
        video = np.load(os.path.join(self.dataroot, name))
        sequence = self.get_sequence(video, int(start_idx), int(end_idx))
        video = self.preprocess(sequence)
        tokens = name.split('.')[0].split('_')
        action = self.action_set.index(tokens[1])
        actor = self.actor_set.index(tokens[0])
        return video, action, actor, name.split('.')[0]

    def __len__(self):
        return len(self.lines)

    def name(self):
        return 'MUGDataset'


class MUGDataset_2(MUGDataset):
    def load_cache(self):
        cache = os.path.join(self.dataroot, 'mug2_cache_%s_%d.db' % (self.mode, self.video_len * self.every_nth))
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.lines, self.lengths, self.actor_set, self.action_set = pickle.load(f)
        else:
            self.lines, self.lengths, self.actor_set, self.action_set = self.build_dataset()
            makedir(self.textroot)
            text_file = os.path.join(self.textroot, 'mug2_%s_list_%d.txt' % (self.mode, self.video_len * self.every_nth))
            with open(text_file, 'w+') as f:
                f.writelines(self.lines)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.lines, self.lengths, self.actor_set, self.action_set), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def get_actor(self, video_path):
        video_name = video_path.split('/')[-1].split('.')[0]
        actor = int(video_name.split('_')[0])
        return actor

    def build_dataset(self):
        lines = []
        lengths = []
        action_set = ['anger', 'disgust', 'surprise', 'sadness', 'happiness', 'fear']
        actors = []
        for video in glob.glob(os.path.join(self.dataroot, '*.npy')):
            actor = self.get_actor(video)
            if actor not in actors:
                actors.append(actor)
        actors = np.sort(actors).tolist()
        num_actor = len(actors)
        actor_set = ['%03d' % actor for actor in actors]
        if self.mode == 'Train':
            actor_set = actor_set[:int(num_actor * 0.7)]
        elif self.mode == 'Test':
            actor_set = actor_set[int(num_actor * 0.7):]
        else:
            raise NotImplementedError('mode %s is not implemented' % self.mode)

        for video in tqdm.tqdm(glob.glob(os.path.join(self.dataroot, '*.npy'))):
            video_name = video.split('/')[-1]
            # <actor>_<action>_<seq_id>_<split_id>.npy
            video_prefix = video_name.split('.')[0]
            tokens = video_prefix.split('_')
            if tokens[0] in actor_set:
                if tokens[1] not in action_set:
                    action_set.append(tokens[1])
                npy_file = np.load(video)
                total_len = len(npy_file)
                if total_len > (self.video_len - 1) * self.every_nth + 1:
                    lines.append('%s %d %d \n' % (video_prefix, 0, total_len - 1))
                    lengths.append(total_len)

        return lines, lengths, actor_set, action_set


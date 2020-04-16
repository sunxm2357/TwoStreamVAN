import sys
sys.path.insert(0, '..')
import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
import cv2
from utils.util import makedir


class WeizmannDataset(torch.utils.data.Dataset):
    """
    Dataloader for the Weizmann Dataset
    """
    def __init__(self, dataroot, textroot, video_len, image_size, every_nth, crop, mode="Train", mini_clip=False):
        """
        Prepare for the data list
        :param dataroot: str, the path for the stored data
        :param textroot: str, the path to write the data list
        :param video_len: int, the length of the generated video
        :param image_size: int, the spatial size of the image
        :param every_nth: int, the frequency to sample frames
        :param crop: bool, true if random cropping
        :param mode: ['Train', 'Test']
        :param mini_clip: bool, specify if the origin video is divided into mini-clips
        """
        print(self.name())

        # parse the args
        self.dataroot = dataroot
        self.textroot = textroot
        self.video_len = video_len
        self.every_nth = every_nth
        self.crop = crop
        self.mode = mode
        self.action_set = ['bend', 'jack', 'pjump', 'wave1', 'wave2', 'jump', 'run', 'side', 'skip', 'walk']
        self.actor_set = ['daria', 'denis', 'eli', 'ido', 'ira', 'lena', 'lyova', 'moshe', 'shahar']
        self.image_size = image_size
        self.mini_clip = mini_clip

        # get the cache name
        if mini_clip:
            cache = os.path.join(self.dataroot, 'cache_mini_%s_%d.db' % (mode, video_len * every_nth))
        else:
            cache = os.path.join(self.dataroot, 'cache_%s_%d.db' % (mode, video_len * every_nth))

        # read the cache file or build the cache file
        if cache is not None and os.path.exists(cache):
            # read the cache file
            with open(cache, 'rb') as f:
                self.lines, self.lengths = pickle.load(f)
        else:
            # build the list
            self.lines, self.lengths = self.build_dataset()

            # write the readable text file
            makedir(textroot)
            if mini_clip:
                text_file = os.path.join(textroot, 'miniclip_%s_list_%d.txt' % (mode, video_len * every_nth))
            else:
                text_file = os.path.join(textroot, '%s_list_%d.txt' % (mode, video_len * every_nth))
            with open(text_file, 'w+') as f:
                f.writelines(self.lines)

            # write the cache file
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.lines, self.lengths), f)

        # get the total frames
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def build_dataset(self):
        """
        Get the {name, start_idx, end_idx} and the length of each video
        :return: lists of {name, start_idx, end_idx} and lengths of videos
        """
        lines = []
        lengths = []

        for video in tqdm.tqdm(os.listdir(self.dataroot)):
            if video.endswith('.npy'):
                # get the every_nth based on the action category
                action = video.split('.')[0].split('_')[1]
                if action in ['run', 'side', 'skip']:
                    every_nth = 1
                else:
                    every_nth = self.every_nth

                # load the npy file
                video_path = os.path.join(self.dataroot, video)
                npy_file = np.load(video_path)
                total_len = npy_file.shape[-1]

                # get all the info based the loading mode
                if self.mode == 'Test':
                    # start_idx and end_idx are inclusive
                    start_idx = total_len * 2 // 3
                    if (total_len - start_idx) <= (self.video_len - 1) * every_nth + 1:
                        start_idx = total_len - ((self.video_len - 1) * every_nth + 1) - 1
                    lines.append('%s %d %d \n' % (video.split('.')[0], start_idx, total_len - 1))
                    lengths.append(total_len - start_idx)
                else:
                    if self.mini_clip:
                        new_lines, new_lengths = self.get_train_samples_miniclip(video, total_len, every_nth)
                    else:
                        new_lines, new_lengths = self.get_train_samples(video, total_len, every_nth)
                    lines += new_lines
                    lengths += new_lengths
        return lines, lengths

    def get_train_samples(self, video_name, total_len, every_nth):
        """
        construct the {name, start_idx, end_idx} and length in training when mini-clip is False
        :param video_name: str, the video name
        :param total_len: int, the length of the original video
        :param every_nth: int, the frequency to sample frames
        :return: line and length lists
        """
        start_idx = 0
        end_idx = total_len * 2 // 3 - 1
        if (end_idx - start_idx) > (self.video_len - 1) * every_nth + 1:
            line = '%s %d %d \n' % (video_name.split('.')[0], start_idx, end_idx)
            length = end_idx - start_idx + 1
            return [line], [length]
        else:
            return [], []

    def get_train_samples_miniclip(self,video_name, total_len, every_nth):
        """
        construct the {name, start_idx, end_idx} and length in training when mini-clip is True
        :param video_name: str, the video name
        :param total_len: int, the length of the original video
        :param every_nth: int, the frequency to sample frames
        :return: line and length lists
        """
        num_miniclips = 20
        start_idx = 0
        end_idx = total_len * 2 // 3 - 1
        last_start = end_idx - (self.video_len - 1) * every_nth
        starts = np.linspace(start_idx, last_start, num_miniclips).astype('int').tolist()
        lines = [('%s %d %d \n' % (video_name.split('.')[0], x, x + (self.video_len - 1) * every_nth)) for x in
                 starts]
        lengths = [(self.video_len - 1) * every_nth + 1] * num_miniclips
        return lines, lengths

    def get_sequence(self, video, start_idx, end_idx):
        """
        Construct the sequence from the origin video
        :param video: ndarray, with size [h, w, c, t]
        :param start_idx: int, the start index
        :param end_idx: int, the end index
        :return: sequence, ndarray, with size [h, w, c, end-start+1]
        """
        picked = range(start_idx, end_idx + 1)
        sequence = video[:, :, :, picked]
        return sequence

    def preprocess(self, sequence):
        """
        crop and resize
        :param sequence: ndarray, with size [h, w, c, end_idx - start_idx +1]
        :return output: ndarray, with size [h, w*(end-start+1), c]
        """
        height, width, _, video_len = sequence.shape
        h, w = height - 5, width - 5
        # random or central crop
        if self.crop:
            tl_h = np.random.randint(low=0, high=5)
            tl_w = np.random.randint(low=0, high=5)
        else:
            tl_h = 2
            tl_w = 2
        cropped_seq = sequence[tl_h: tl_h + h, tl_w: tl_w + w]

        # resize
        frames = np.split(cropped_seq, video_len, axis=-1)
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
        return 'WeizmannDataset'

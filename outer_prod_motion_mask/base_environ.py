import sys
sys.path.insert(0, '..')

import os
import time

import numpy as np

import torch
from torch.autograd import Variable
import torchvision.utils as vutils
from torch import nn
import torch.nn.init as init

from abc import ABCMeta, abstractmethod
from tensorboardX import SummaryWriter
from utils.util import print_current_errors, makedir, Initializer
import cv2

import pdb

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class BaseEnviron(object):
    __metaclass__ = ABCMeta

    def __init__(self, gen_args, checkpoint_dir, log_dir, output_dir, video_len, action_set, actor_set, is_eval=False):
        """
        :param gen_args: dict, all the parameters/settings for the generator network
        :param checkpoint_dir:  str, the path to save/load checkpoints
        :param log_dir:  str, the path to save the log file
        :param output_dir: str, the path to save the generated videos
        :param video_len: str, the desired length of the generated videos
        :param action_set: list, the action set
        :param actor_set: list, the actor set
        :param is_eval: bool, specify for evaluation
        """

        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.networks = {}

        self.num_categories = gen_args['num_categories']
        self.video_len = video_len
        self.action_set = action_set
        self.actor_set = actor_set
        self.is_eval = is_eval

        if not is_eval:
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.optimizers = {}
            # define loss
            self.losses = {}
            self.mse_loss = nn.MSELoss()
            self.gan_loss = nn.BCEWithLogitsLoss()
            self.category_criterion = nn.CrossEntropyLoss()

    def move_to_gpu(self):
        """
        Move the network to GPU when it is available
        """
        for k, v in self.networks.items():
            self.networks[k].cuda()

    def weight_init(self):
        """
        The initialization method of the network
        """
        for key in self.networks.keys():
            Initializer.initialize(model=self.networks[key], initialization=init.xavier_uniform_,
                                   gain=init.calculate_gain('relu'))

    def set_inputs(self, batch):
        """
        :param batch: {'images': a tensor [batch_size, c, video_len, h, w],
                        'categories': np.ndarray [batch_size,],
                        'actor': np.ndarray [batch_size,]}
        """
        if torch.cuda.is_available():
            self.images = Variable(batch['images']).cuda()
            self.categories = Variable(batch['categories']).cuda()
            self.actors = Variable(batch['actors'].cuda())
        else:
            self.images = Variable(batch['images'])
            self.categories = Variable(batch['categories'])
            self.actors = Variable(batch['actors'])

    def get_category(self, cls_id=None, batch_size=32):
        """
        get category variable for the specific class or random classes
        :param cls_id: int, specify if the video belongs to the certain classes
        :param batch_size: int, the batch size
        """
        if cls_id is None:
            num = batch_size // self.num_categories
            random_labels = np.expand_dims(np.arange(self.num_categories), axis=0).repeat(num, 1).reshape(-1)
            offset = batch_size % self.num_categories
            offset_labels = np.random.randint(low=0, high=self.num_categories - 1, size=[offset, ])
            random_labels = np.concatenate((random_labels, offset_labels), axis=0)
        else:
            random_labels = cls_id * np.ones([batch_size, ]).astype('int')
        if torch.cuda.is_available():
            self.categories = Variable(torch.from_numpy(random_labels)).cuda()
        else:
            self.categories = Variable(torch.from_numpy(random_labels))
        self.first_img = None
        self.images = None

    # ************************************************************************
    # ************************** Backward Methods ****************************
    # ************************************************************************

    def backward_content_enc(self, retain):
        """
        Backward method for the content encoder
        :param retain: bool, whether to retain the graph after the update
        """
        self.optimizers['optimize_contEnc'].zero_grad()
        self.losses['cont_enc']['l_enc_recons'].backward(retain_graph=retain)
        self.optimizers['optimize_contEnc'].step()

    def backward_motion_enc(self, retain):
        """
        Backward method for the motion encoder
        :param retain: bool, whether to retain the graph after the update
        """
        self.optimizers['optimize_motionEnc'].zero_grad()
        if self.task == 'pred':
            self.losses['motion_enc']['l_enc_pred'].backward(retain_graph=retain)
        elif self.task == 'video_generate':
            self.losses['vid_motion_enc']['l_enc'].backward(retain_graph=retain)
        else:
            raise ValueError('task %s is not supported' % self.task)
        self.optimizers['optimize_motionEnc'].step()

    def backward_img_dis(self):
        """
        Backward method for the image discriminator
        """
        self.optimizers['optimize_d_img'].zero_grad()
        if self.task == 'pred':
            self.losses['img_dis']['l_dis_pred'].backward()
        elif self.task == 'recons':
            self.losses['img_dis']['l_dis_recons'].backward()
        else:
            raise ValueError('task %s is not supported' % self.task)
        self.optimizers['optimize_d_img'].step()

    def backward_vid_dis(self):
        """
        Backward method for the video discriminator
        """
        self.optimizers['optimize_d_vid'].zero_grad()
        self.losses['vid_dis']['l_dis'].backward()
        self.optimizers['optimize_d_vid'].step()

    # ************************************************************************
    # ************************** Visualization *******************************
    # ************************************************************************
    def get_visualizations(self):
        """
        Get the visualization components of different learning task
        """
        if self.task == 'recons':
            inputs, targets, x_tilde, x_p = self.recons_img, self.recons_img, self.recons_x_tilde, self.recons_x_p
        elif self.task == 'pred':
            inputs, targets, x_tilde, x_p = self.prev_img, self.pred_target, self.pred_x_tilde, self.pred_x_p
        else:
            raise ValueError('mode %s is not supported in get_visualizations' % self.task)
        return inputs, targets, x_tilde, x_p

    def make_grid_frame(self):
        """
        make frame grid in form of
                gt      gt      gt      ... gt
                target  target  target  ... target
                x_tilde x_tilde x_tilde ... x_tilde
                x_p     x_p     x_p     ... x_p
                ....
        """
        inputs, targets, x_tilde, x_p = self.get_visualizations()
        batch_size = int(self.images.shape[0])
        nrow = int(batch_size / 4)
        visuals = []
        for start_idx in range(0, batch_size, nrow):
            visual_inputs = vutils.make_grid(inputs[start_idx: start_idx + nrow], nrow=nrow)
            visual_targets = vutils.make_grid(targets[start_idx: start_idx + nrow], nrow=nrow)
            visual_xtilde = vutils.make_grid(x_tilde[start_idx: start_idx + nrow], nrow=nrow)
            visual_xp = vutils.make_grid(x_p[start_idx: start_idx + nrow], nrow=nrow)
            visuals += [visual_inputs, visual_targets, visual_xtilde, visual_xp]
        grid = torch.cat(visuals, dim=1)
        grid = (grid + 1) / 2
        return grid

    def make_grid_video(self):
        """
        make video grid in form of
                gt_1      gt_2      gt_3      ... gt_n
                x_tilde_1 x_tilde_2 x_tilde_3 ... x_tilde_n
                x_p_1     x_p_2     x_p_3     ... x_p_n
                ....
        """
        c, h, w = int(self.images.shape[1]), int(self.images.shape[3]), int(self.images.shape[4])
        visuals = torch.stack((self.images, self.video_x_tilde, self.video_x_p), dim=1).permute(0, 1, 3, 2, 4,
                                                                                                5).contiguous().view(
            -1, c, h, w)
        grid = vutils.make_grid(visuals, nrow=self.video_len)
        grid = (grid + 1) / 2
        return grid

    def visual_batch(self, current_iter, name='current batch'):
        """
        get the visualization grid for different tasks: frame-level task or video-level task
        containing the input, (target), ae_output and gan_output of the current batch
        """

        if self.task == 'video_generate':
            # video-level task
            grid = self.make_grid_video()
        else:
            # frame-level task
            grid = self.make_grid_frame()

        # restrict to (0,1)
        grid = torch.clamp(grid, 0, 1)

        self.writer.add_image(name, grid, current_iter)

    def save_batch(self, current_iter, names=None, start_idx=0):
        """
        save the batch for the generation conditioned on the first frame
        :param current_iter: int, the current iteration
        :param names: the name of videos where the first frame is from
        :param start_idx: int, the start index of the current batch
        :return: output_dir: the path of the output folder
        """
        output_dir = os.path.join(self.output_dir, 'evaluation', str(current_iter))
        makedir(output_dir)

        video = self.video_x_p.cpu().data.numpy().transpose((0, 2, 1, 3, 4))
        self.save_video(video, output_dir, self.categories.cpu().data.numpy(), self.actors.cpu().data.numpy(),
                        names=names, start_idx=start_idx)
        return output_dir

    def save_video(self, video, output_dir, actions, actors=None, names=None, start_idx=0):
        """
        :param video: ndarray [batch_size, video_len, c, h, w]
        :param output_dir: str, the path of the output folder
        :param actions: ndarray [batch_size, ], the input actions for each video
        :param actors: ndarray [batch_size, ], specify if the generation is provided with the first frame
        :param names: list of str, the file name of each video
        :param start_idx: int, the start index of the current batch
        """
        batch_size = video.shape[0]
        videos = np.split(video, batch_size, axis=0)
        for idx, sample in enumerate(videos):
            if actors is None:
                prefix = 'none_%s' % self.action_set[actions[idx]]
                npy_path = os.path.join(output_dir, '%s_%03d.npy' % (prefix, start_idx + idx))
            else:
                if names is None:
                    prefix = '%s_%s' % (self.actor_set[actors[idx]], self.action_set[actions[idx]])
                else:
                    prefix = names[idx]
                npy_path = os.path.join(output_dir, '%s_%03d.npy' % (prefix, start_idx))
            np.save(npy_path, sample.squeeze())

    def get_loss_dict(self):
        """
        Get the loss dictionary; extract the loss value from the tensor
        :return: the loss dictionary
        """
        loss = {}
        for key in self.losses.keys():
            loss[key] = {}
            for subkey, v in self.losses[key].items():
                if not isinstance(v, int) and not isinstance(v, float):
                    loss[key][subkey] = v.data
        return loss

    def print_loss(self, current_iter, start_time, metrics=None):
        """
        Print the loss on the screen
        :param current_iter: int, the current iteration
        :param start_time: time, the start time of the current iteration
        :param metrics: dictionary when printing the validation metrics
        """
        if metrics is None:
            loss = self.get_loss_dict()
        else:
            loss = {'metrics': metrics}

        print('-------------------------------------------------------------')
        for key in loss.keys():
            for subkey in loss[key].keys():
                self.writer.add_scalar('%s/%s'%(key, subkey), loss[key][subkey], current_iter)
            print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter, loss[key],
                                 time.time() - start_time)

    # ************************************************************************
    # **************************** Save and Load *****************************
    # ************************************************************************

    def get_current_state(self, current_iter):
        """
        Get the current state of networks and optimizers
        :param current_iter: int, the current iteration
        :return: the dictionary of the current state
        """
        current_state = {}
        for k, v in self.networks.items():
            current_state[k] = v.state_dict()
        for k, v in self.optimizers.items():
            current_state[k] = v.state_dict()
        current_state['iter'] = current_iter
        return current_state

    def save(self, label, current_iter):
        """
        save the checkpoints of the current iteration
        :param label: str, the name of the checkpoint
        :param current_iter: int, the current iteration
        """
        current_state = self.get_current_state(current_iter)
        save_filename = '%s_model.pth.tar' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(current_state, save_path)

    def load_snapshot(self, snapshot, is_eval=False):
        """
        Load the params of the networks and the optimizers (during the training)
        :param snapshot: the dictionary of the saved state
        :param is_eval: bool, specify for the evaluation
        :return: return the current iteration
        """
        for k, v in self.networks.items():
            if k in snapshot.keys():
                model_dict = v.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {kk: vv for kk, vv in snapshot[k].items() if kk in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.networks[k].load_state_dict(model_dict)
        if not is_eval:
            for k, v in self.optimizers.items():
                if k in snapshot.keys():
                    self.optimizers[k].load_state_dict(snapshot[k])
        return snapshot['iter']

    def load(self, label, is_eval=False):
        """
        Load the saved checkpoints
        :param label: str, the name of the checkpoints
        :param is_eval: bool, specify during the evaluation
        """
        save_filename = '%s_model.pth.tar' % label
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            snapshot = torch.load(save_path)
            return self.load_snapshot(snapshot, is_eval=is_eval)
        else:
            raise ValueError('snapshot %s does not exist' % save_path)

    # ************************************************************************
    # **************************** Network Mode ******************************
    # ************************************************************************

    def train(self):
        """
        Move all components in the network to the training mode
        """
        for key in self.networks.keys():
            self.networks[key].train()

    def eval(self):
        """
        Move all components in the network to the evaluation mode
        """
        for key in self.networks.keys():
            self.networks[key].eval()

    # ************************************************************************
    # **************************** Static Method *****************************
    # ************************************************************************
    @staticmethod
    def kl_loss(pair):
        """
        :param pair: a tuple of mean and variance of a Gaussian distribution
        :return: the KL divergence between the input Gaussian and N(0,1)
        """
        mean = pair[0]
        logvar = pair[1]
        var = torch.exp(logvar)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mean**2 - var, dim=-1), dim=0)

    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)


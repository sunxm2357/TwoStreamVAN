import sys
sys.path.insert(0, '..')
import torch
from torch import nn
from torch.autograd import Variable
from utils.util import one_hot
import numpy as np
import pdb
import torch.nn.functional as F
import random
from copy import deepcopy


class GDL(nn.Module):
    """Image gradient difference loss as defined by Mathieu et al. (https://arxiv.org/abs/1511.05440)."""

    def __init__(self, c_dim):
        """Constructor
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        """
        super(GDL, self).__init__()
        self.loss = nn.L1Loss()
        self.filter_w = np.zeros([c_dim, c_dim, 1, 2])
        self.filter_h = np.zeros([c_dim, c_dim, 2, 1])
        for i in range(c_dim):
            self.filter_w[i, i, :, :] = np.array([[-1, 1]])
            self.filter_h[i, i, :, :] = np.array([[1], [-1]])

    def forward(self, output, target):
        """Forward method
        :param output: The predicted output
        :param target: The desired output
        """
        filter_w = Variable(torch.from_numpy(self.filter_w).float().cuda())
        filter_h = Variable(torch.from_numpy(self.filter_h).float().cuda())
        output_w = F.conv2d(output, filter_w, padding=(0, 1))
        output_h = F.conv2d(output, filter_h, padding=(1, 0))
        target_w = F.conv2d(target, filter_w, padding=(0, 1))
        target_h = F.conv2d(target, filter_h, padding=(1, 0))
        return self.loss(output_w, target_w) + self.loss(output_h, target_h)


class ImgEnc(nn.Module):
    """
    Content Encoder for the single frame

    """
    def __init__(self, input_dim, gf_dim=16, use_bn=False):
        """
        :param input_dim: int, the number of color channels for the input image
        :param gf_dim: int, the base dimension of the CNN
        :param use_bn: bool, specify if using batch normalization
        """
        super(ImgEnc, self).__init__()
        self.channels = input_dim
        self.gf_dim = gf_dim
        # CNN layers
        conv1 = nn.Conv2d(input_dim, gf_dim, kernel_size=3, stride=1, padding=1, bias=False)
        relu1 = nn.ReLU()

        conv2 = nn.Conv2d(gf_dim, gf_dim, kernel_size=3, stride=2, padding=1, bias=False)
        relu2 = nn.ReLU()

        conv3 = nn.Conv2d(gf_dim, gf_dim, kernel_size=3, stride=1, padding=1, bias=False)
        relu3 = nn.ReLU()

        conv4 = nn.Conv2d(gf_dim, 2*gf_dim, kernel_size=3, stride=2, padding=1, bias=False)
        relu4 = nn.ReLU()

        conv5 = nn.Conv2d(2*gf_dim, 2*gf_dim, kernel_size=3, stride=1, padding=1, bias=False)
        relu5 = nn.ReLU()

        conv6 = nn.Conv2d(2*gf_dim, gf_dim, kernel_size=3, stride=2, padding=1, bias=False)
        relu6 = nn.ReLU()

        # fc_layers:
        fc1 = nn.Linear(8 * 8 * gf_dim, gf_dim * 32)
        fc_relu1 = nn.ReLU()

        if use_bn:
            # BN layers for CNN
            bn1 = nn.BatchNorm2d(gf_dim)
            bn2 = nn.BatchNorm2d(gf_dim)
            bn3 = nn.BatchNorm2d(gf_dim)
            bn4 = nn.BatchNorm2d(2*gf_dim)
            bn5 = nn.BatchNorm2d(2*gf_dim)
            bn6 = nn.BatchNorm2d(gf_dim)

            # BN layers for FC
            fc_bn1 = nn.BatchNorm1d(gf_dim * 32)

            self.encode_cnn = nn.Sequential(conv1, bn1, relu1,
                                            conv2, bn2, relu2,
                                            conv3, bn3, relu3,
                                            conv4, bn4, relu4,
                                            conv5, bn5, relu5,
                                            conv6, bn6, relu6)
            self.fc1_block = nn.Sequential(fc1, fc_bn1, fc_relu1)

        else:
            self.encode_cnn = nn.Sequential(conv1, relu1,
                                            conv2, relu2,
                                            conv3, relu3,
                                            conv4, relu4,
                                            conv5, relu5,
                                            conv6, relu6)
            self.fc1_block = nn.Sequential(fc1, fc_relu1)

    def forward(self, input):
        """Forward method
        :param input: A single frame [batch_size, input_dim, h, w]
        :return: mean and variance for z_motion [batch_size, dim_z_motion]
        """
        inter_result = self.encode_cnn(input).view(-1, 8 * 8 * self.gf_dim)
        output = self.fc1_block(inter_result)
        return output


class MotionEnc(nn.Module):
    """The motion encoder as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).
    This module takes a difference frame and produces an encoded representation with reduced resolution. It also
    produces the intermediate convolutional activations for use with residual layers.
    """

    def __init__(self, input_dim, gf_dim=16):
        """Constructor
        :param input_dim: int, the number of color channels for the input image
        :param gf_dim:  int, the base dimension of the CNN
        """
        super(MotionEnc, self).__init__()
        conv1 = nn.Conv2d(input_dim, gf_dim, 5, padding=2)
        bn1 = nn.BatchNorm2d(gf_dim)
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d(2)

        conv2 = nn.Conv2d(gf_dim, gf_dim * 2, 5, padding=2)
        bn2 = nn.BatchNorm2d(gf_dim * 2)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d(2)

        conv3 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 7, padding=3)
        bn3 = nn.BatchNorm2d(gf_dim * 4)
        relu3 = nn.ReLU()
        pool3 = nn.MaxPool2d(2)

        conv4 = nn.Conv2d(gf_dim * 4, gf_dim * 4, 7, padding=3)
        bn4 = nn.BatchNorm2d(gf_dim * 4)
        relu4 = nn.ReLU()
        pool4 = nn.MaxPool2d(2)

        conv5 = nn.Conv2d(gf_dim * 4, gf_dim * 8, 4)
        bn5 = nn.BatchNorm2d(gf_dim * 8)
        relu5 = nn.ReLU()

        fc1 = nn.Linear(gf_dim * 8, 128)
        fc_bn1 = nn.BatchNorm1d(gf_dim * 8)
        fc_relu1 = nn.ReLU()

        self.main = nn.Sequential(conv1, bn1, relu1, pool1,
                                  conv2, bn2, relu2, pool2,
                                  conv3, bn3, relu3, pool3,
                                  conv4, bn4, relu4, pool4,
                                  conv5, bn5, relu5)
        self.fc_block = nn.Sequential(fc1, fc_bn1, fc_relu1)

    def forward(self, input_diff):
        """Forward method
        :param input_diff: A difference frame [batch_size, input_dim, h, w]
        :return: the hidden embedding of the difference map after FC layer
        """
        feature = self.main(input_diff)
        output = self.fc_block(feature.squeeze())
        return output


class Sampler(nn.Module):
    """
    Calculate the mean and variance of the approximated latent distribution
    And output a vector from the distribution, which is either sampled from or using the mean of distribution
    """
    def __init__(self, input_dim, h_dim=512):
        """
        Two FC layers to compute the mean and variance
        :param input_dim: int, the dimension of the embedding vector
        :param h_dim: int, the dimension of the latent variable
        """
        super(Sampler, self).__init__()
        self.mean = nn.Linear(input_dim, h_dim)
        self.logvar = nn.Linear(input_dim, h_dim)

    def forward(self, input1, input2, use_mean=False):
        """
        The forward method
        :param input1: the image embedding [batch_size, embedding_dim]
        :param input2: one-hot vector for the action category [batch_size, num_categories],
        where num_categories+embedding_dim = input_dim
        :return: sampled random vector, mean and variance of the constructed distribution
        """
        input = torch.cat((input1, input2), dim=1)
        mean = self.mean(input)
        logvar = self.logvar(input)
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        if use_mean:
            return mean, mean, logvar
        else:
            return eps.mul(std).add_(mean), mean, logvar


class VideoDec(nn.Module):
    """
    The single decoder for the next frame, taking both the content and motion vectors as inputs
    """
    def __init__(self, output_dim, gf_dim=16, use_bn=False):
        """
        :param gf_dim: int, the base dimension of the CNN
        :param output_dim: int, the number of color channels for the output image
        :param use_bn: bool, specify if using batch normalization
        """
        super(VideoDec, self).__init__()
        # CNN layers
        deconv1 = nn.ConvTranspose2d(gf_dim, gf_dim * 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        relu1 = nn.ReLU()

        deconv2 = nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, kernel_size=3, stride=1, padding=1, bias=False)
        relu2 = nn.ReLU()

        deconv3 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        relu3 = nn.ReLU()

        deconv4 = nn.ConvTranspose2d(gf_dim * 2, gf_dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        relu4 = nn.ReLU()

        deconv5 = nn.ConvTranspose2d(gf_dim * 2, gf_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        relu5 = nn.ReLU()

        deconv6 = nn.ConvTranspose2d(gf_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        tanh = nn.Tanh()

        if use_bn:
            bn1 = nn.BatchNorm2d(gf_dim * 8)
            bn2 = nn.BatchNorm2d(gf_dim * 4)
            bn3 = nn.BatchNorm2d(gf_dim * 2)
            bn4 = nn.BatchNorm2d(gf_dim * 2)
            bn5 = nn.BatchNorm2d(gf_dim)
            self.main = nn.Sequential(deconv1, bn1, relu1,
                                      deconv2, bn2, relu2,
                                      deconv3, bn3, relu3,
                                      deconv4, bn4, relu4,
                                      deconv5, bn5, relu5,
                                      deconv6, tanh)
        else:
            self.main = nn.Sequential(deconv1, relu1,
                                      deconv2, relu2,
                                      deconv3, relu3,
                                      deconv4, relu4,
                                      deconv5, relu5,
                                      deconv6, tanh)

    def forward(self, input):
        """
        :param input: content input [batch_size. gf_dim. 1, 1]
        :return: a tensor [batch_size, output_dim, h, w]
        """
        output = self.main(input)
        return output


class TrajGenerator(nn.Module):
    """
    Adopting LSTM to encode the sequence history conditioned on the action class
    """
    def __init__(self, h_dim, num_categories, output_dim):
        """
        :param h_dim: int, the dimension of the latent space
        :param num_categories: int, the number of the categories
        :param output_dim: int, the output dimension
        """
        super(TrajGenerator, self).__init__()
        self.h_dim = h_dim
        self.num_categories = num_categories
        self.output_dim = output_dim

        fc = nn.Linear(h_dim + num_categories, 128)
        fc_bn = nn.BatchNorm1d(128)
        relu = nn.ReLU()

        self.fc_block = nn.Sequential(fc, fc_bn, relu)
        self.main = nn.LSTM(input_size=128, hidden_size=output_dim, batch_first=True)

    def get_initial_state(self, batch_size):
        """
        Create the initial state of the LSTM, as the input for the one-step update

        :param batch_size: int, the size of the training batch
        :return: the tuple of current state (h, c)
        """
        h_0 = Variable(torch.zeros((1, batch_size, self.output_dim)))
        c_0 = Variable(torch.zeros((1, batch_size, self.output_dim)))
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        current_state = (h_0, c_0)
        return current_state

    def one_step(self, motion_h, one_hot_matrix, state_0):
        """
        Update the LSTM for one timestep
        :param motion_h: the hidden embedding of the motion with size [batch, h_dim]
        :param one_hot_matrix: the one-hot vector of the current batch with size [batch, num_categories]
        :param state_0: the tuple of the LSTM state before this time step, (h_0, c_0)
        :return: state_1: the tuple of the LSTM state after this time step, (h_1, c_1)
        """
        input_fc = torch.cat((motion_h, one_hot_matrix), dim=-1)
        output_fc = self.fc_block(input_fc).view(-1, 1, self.output_dim)
        output, state_1 = self.main(output_fc, state_0)
        return output, state_1

    def forward(self, motion_h, one_hot_matrix):
        """
        Update the LSTM for multiple timesteps
        :param motion_h: Variable of a tensor [batch_size, seq_len, 128]
        :param one_hot_matrix: Variable of a tensor [batch_size, num_categories]
        :return: output: Variable of a tensor [batch_size, seq_len, output_dim]
        """
        seq_len = int(motion_h.shape[1])
        one_hot_matrix_tile = one_hot_matrix.unsqueeze(1).repeat(1, seq_len, 1)
        input_fc = torch.cat((motion_h, one_hot_matrix_tile), dim=-1).view(-1, self.h_dim + self.num_categories)
        output_fc = self.fc_block(input_fc).view(-1, seq_len, self.output_dim)
        output, current_state = self.main(output_fc)
        return output


class CombLayer(nn.Module):
    """
    Condition the content latent vector on the action class;
    Then combine the content and motion latent vectors
    """
    def __init__(self, content_dim, motion_dim, num_categories, gf_dim=16, use_bn=False):
        """
        :param content_dim: int, the dimension of the content latent vector
        :param motion_dim: int, the dimension of the motion latent vector
        :param num_categories: int, the number of categories
        :param gf_dim: int, the base number of channels
        :param use_bn: bool, specify if using the batch normalization
        """
        super(CombLayer, self).__init__()
        self.gf_dim = gf_dim

        # condition the content vector on the action categories
        fc1 = nn.Linear(content_dim + num_categories, 512)
        fc_relu1 = nn.ReLU()

        # combine the motion and content
        fc2 = nn.Linear(512 + motion_dim, 512)
        fc_relu2 = nn.ReLU()
        fc3 = nn.Linear(512, 8 * 8 * gf_dim)
        fc_relu3 = nn.ReLU()

        if use_bn:
            fc_bn1 = nn.BatchNorm1d(512)
            fc_bn2 = nn.BatchNorm1d(512)
            fc_bn3 = nn.BatchNorm1d(8 * 8 * gf_dim)
            self.fc_block1 = nn.Sequential(fc1, fc_bn1, fc_relu1)
            self.fc_block2 = nn.Sequential(fc2, fc_bn2, fc_relu2, fc3, fc_bn3, fc_relu3)
        else:
            self.fc_block1 = nn.Sequential(fc1, fc_relu1)
            self.fc_block2 = nn.Sequential(fc2, fc_relu2, fc3, fc_relu3)

    def forward(self, cont_input, motion_input, one_hot_matrix):
        """
        :param cont_input: the content latent variable with the size [batch, content_dim]
        :param motion_input: the motion latent variable with the size [batch, motion_dim]
        :param one_hot_matrix: the one-hot vector [batch, num_categories]
        :return: the content-motion representation [batch, gf_dim, 8, 8]
        """
        input_fc1 = torch.cat((cont_input, one_hot_matrix), dim=1)
        output_fc1 = self.fc_block1(input_fc1)
        input_fc2 = torch.cat((output_fc1, motion_input), dim=1)
        output_fc2 = self.fc_block2(input_fc2).view(-1, self.gf_dim, 8, 8)
        return output_fc2


class Generator(nn.Module):
    """
    Generative part of the GAN model
    """
    def __init__(self, num_categories, n_channels, motion_dim=128, cont_dim=512, gf_dim=32, use_bn=False):
        """
        :param num_categories: int, the number of action categories
        :param n_channels: int, the number of color channels of the image
        :param motion_dim: int, the dimension of the motion latent space
        :param cont_dim: int, the dimension of the content latent space
        :param gf_dim: int, the base number of channels
        :param use_bn: bool, specify if using the batch normalization
        """
        super(Generator, self).__init__()
        self.num_categories = num_categories
        self.cont_dim = cont_dim
        self.motion_dim = motion_dim

        self.contEnc = ImgEnc(n_channels, gf_dim=gf_dim, use_bn=use_bn)
        self.contSampler = Sampler(gf_dim * 32 + num_categories, h_dim=cont_dim)
        self.motionEnc = MotionEnc(n_channels)
        self.motionSampler = Sampler(128 + num_categories, h_dim=motion_dim)
        self.trajGenerator = TrajGenerator(motion_dim, self.num_categories, 128)
        self.combLayer = CombLayer(cont_dim, 128, num_categories, gf_dim=gf_dim, use_bn=use_bn)
        self.videoDec = VideoDec(n_channels, gf_dim=gf_dim, use_bn=use_bn)

    @staticmethod
    def get_rand_var(batch_size, dim):
        z = torch.randn((batch_size, dim))
        if torch.cuda.is_available():
            z = Variable(z.cuda())
        else:
            z = Variable(z)
        return z

    @staticmethod
    def get_zero_var(batch_size, dim):
        m = torch.zeros((batch_size, dim))
        if torch.cuda.is_available():
            m = Variable(m.cuda())
        else:
            m = Variable(m)
        return m

    def reconstruct_one_frame(self, cat, input_frame=None, mode='sample'):
        """
        Content Stream: generate or reconstruct a single frame
        :param input_frame: Variable of a tensor [batch_size, c, h, w]
        :param cat: Variable of a tensor [batch_size]
        :param mode: 'sample' for sampling from the latent space;
                     'mean' for picking the mean of the distribution;
                     'random' for sampling from the standard normal distribution N(0, 1)
        :return: reconstructed_frame, mean, logvar for the content latent space
        """
        # calculate one_hot vector from cat
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))
        batch_size = int(cat.shape[0])

        if mode == 'random':
            z_cont = self.get_rand_var(batch_size, self.cont_dim)
            mean, logvar = None, None
        else:
            cont_h = self.contEnc(input_frame)
            if mode == 'sample':
                z_cont, mean, logvar = self.contSampler(cont_h, one_hot_matrix)
            elif mode == 'mean':
                z_cont, mean, logvar = self.contSampler(cont_h, one_hot_matrix, use_mean=True)
            else:
                raise ValueError('mode %s is not supported' % mode)

        motion_state = self.get_zero_var(batch_size, 128)
        comb_feat = self.combLayer(z_cont, motion_state, one_hot_matrix)
        frame = self.videoDec(comb_feat)
        return frame, mean, logvar

    def predict_next_frame(self, prev_frame, diff_frames, timestep, cat, mode='sample'):
        """
        The simpler task for motion stream -- predict the next frame given the prev_frame, diff_frames
        :param prev_frame: Variable of a tensor [batch_size, c, h, w]
        :param diff_frames: Variable of a tensor [batch_size, c, video_len-1, h, w]
        :param timestep: int which indicates which time step to predict
        :param cat: Variable of a tensor [batch_size]
        :param mode:  'sample' for sampling from the latent space;
                      'mean' for picking the mean of the distribution;
                      'random' for sampling from the standard normal distribution N(0, 1)
        :return: predicted_frame, mean, logvar for the motion latent space
        """
        # prepare one_hot vector
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))
        batch_size, c,  h, w = int(prev_frame.shape[0]), int(prev_frame.shape[1]), int(prev_frame.shape[2]), int(prev_frame.shape[3])

        # get the content feature
        cont_h = self.contEnc(prev_frame)
        z_cont, _, _ = self.contSampler(cont_h, one_hot_matrix, use_mean=True)

        # build the batch of the difference maps
        if timestep == 1:
            motion_input = diff_frames[:, :, timestep - 1]
        else:
            # [batch, c, t, h, w] - > [batch, t, c, h, w] -> [batch * t, c, h, w]
            motion_input = diff_frames[:, :, :timestep].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

        # motion_h: [batch * t, 128] -> [batch, t, 128]
        motion_h = self.motionEnc(motion_input).view(-1, timestep, 128)

        if timestep == 1:
            # if predicting the 2nd frame in the sequence
            if mode == 'random':
                z_motion = self.get_rand_var(batch_size, self.motion_dim).view(-1, 1, self.motion_dim)
                mean, logvar = None, None
            else:
                if mode == 'sample':
                    z_motion, mean, logvar = self.motionSampler(motion_h.squeeze(), one_hot_matrix)
                elif mode == 'mean':
                    z_motion, mean, logvar = self.motionSampler(motion_h.squeeze(), one_hot_matrix, use_mean=True)
                else:
                    raise ValueError('mode %s is not supported' % mode)
                z_motion = z_motion.view(-1, 1, self.motion_dim)
        else:
            # if predicting the frame at other timesteps
            # separate the last difference map and the previous ones

            # generate motion variables for the previous diff maps
            motion_h_prev = motion_h[:, :-1].contiguous().view(-1, 128)
            one_hot_matrix_tile = one_hot_matrix.unsqueeze(1).repeat(1, timestep-1, 1).view(-1, self.num_categories)
            z_motion_prev, _, _ = self.motionSampler(motion_h_prev, one_hot_matrix_tile, use_mean=True)
            z_motion_prev = z_motion_prev.view(-1, timestep-1, self.motion_dim)

            # generate the motion variable for the last diff map
            if mode =='random':
                z_motion_cur = self.get_rand_var(batch_size, self.motion_dim).view(-1, 1, self.motion_dim)
                mean, logvar = None, None
            else:
                motion_h_current = motion_h[:, -1]
                if mode == 'sample':
                    z_motion_cur, mean, logvar = self.motionSampler(motion_h_current, one_hot_matrix)
                elif mode == 'mean':
                    z_motion_cur, mean, logvar = self.motionSampler(motion_h_current, one_hot_matrix, use_mean=True)
                else:
                    raise ValueError('mode %s is not supported' % mode)
                z_motion_cur = z_motion_cur.view(-1, 1, self.motion_dim)
            # stack motion vectors from the previous and the last maps
            z_motion = torch.cat((z_motion_prev, z_motion_cur), dim=1)

        # use the LSTM to encode the motion history
        motion_state = self.trajGenerator(z_motion, one_hot_matrix)[:, -1]

        # combine the motion and content features
        comb_feat = self.combLayer(z_cont, motion_state, one_hot_matrix)

        # decode the next frame
        frame = self.videoDec(comb_feat)

        return frame, mean, logvar

    def full_test(self, cat, video_len):
        """
        :param cat: Variable of a tensor [batch_size]
        :param video_len: the length of the generated video
        :return video: [batch, t, c, h, w]
        """
        frames = []
        batch_size = int(cat.shape[0])
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))

        # generate the first image
        first_frame, _, _ = self.reconstruct_one_frame(cat, mode='random')
        frames.append(first_frame)

        # generate the last motion latent vectors for the generation at each time step
        z_motion = [self.get_rand_var(batch_size, self.motion_dim).view(-1, 1, self.motion_dim) for _ in range(video_len-1)]

        # get initial state of the LSTM
        state = self.trajGenerator.get_initial_state(batch_size)

        for idx in range(video_len - 1):
            # get the content vector of the current frame
            cont_h = self.contEnc(frames[-1])
            z_cont, _, _ = self.contSampler(cont_h, one_hot_matrix, use_mean=True)

            # get motion encoding
            if idx > 0:
                # get the embedding of the last/new difference map to update the LSTM
                current_diff = frames[-2] - frames[-1]
                current_motion_h = self.motionEnc(current_diff)
                motion_h_prev, _, _ = self.motionSampler(current_motion_h, one_hot_matrix, use_mean=True)
                _, state = self.trajGenerator.one_step(motion_h_prev.squeeze(), one_hot_matrix, state)

            # update the LSTM using the sample of the motion space
            motion_state, _ = self.trajGenerator.one_step(z_motion[idx].squeeze(), one_hot_matrix, state)

            # combine the motion and content vectors
            comb_feat = self.combLayer(z_cont, motion_state.squeeze(), one_hot_matrix)

            # decode the next frame
            frame = self.videoDec(comb_feat)
            frames.append(frame)

        video = torch.stack(frames, dim=1)
        return video

    def reconstruct_seq(self, images, cat, diff_frames, epsilon, mode='sample'):
        """
           The final task for motion stream: reconstruct the entire sequence given the first frame and all diff frames
           :param images: the ground-truth video [batch, c, t, h, w]
           :param diff_frames: Variable of a tensor [batch, c, video_len-1, h, w]
           :param cat: Variable of a tensor [batch_size]
           :param epsilon: the probability of the ground truth input in the scheduled sampling
           :param mode:  'sample' for sampling from the motion distribution;
                         'mean' for picking the mean of the motion distribution;
                         'random' for sampling from the standard normal distribution N(0, 1)
           :return: the reconstructed sequence [batch, c, t, h, w], mean, logvar for the motion latent space for all time steps
        """
        frames = []
        batch_size, c, video_len, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2]), \
                                         int(images.shape[3]), int(images.shape[4])
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))

        # reconstruct the first frame
        first_frame = images[:, :, 0]
        first_frame, _, _ = self.reconstruct_one_frame(cat, input_frame=first_frame, mode='mean')
        frames.append(first_frame)

        # get motion embedding
        # get the gt motion encoded vectors
        ### motion_input: [batch, c, t, h, w] -> [batch, t, c, h, w] -> [batch * t, c, h, w]
        motion_input = diff_frames.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        motion_h = self.motionEnc(motion_input)
        one_hot_matrix_tile = one_hot_matrix.unsqueeze(1).repeat(1, video_len - 1, 1).view(-1, self.num_categories)
        z_motion_gt, mean, logvar = self.motionSampler(motion_h, one_hot_matrix_tile)

        motion_gt_mean, _, _ = self.motionSampler(motion_h, one_hot_matrix_tile, use_mean=True)
        motion_gt_mean = motion_gt_mean.view(-1, video_len - 1, self.motion_dim)
        z_motion_gt = z_motion_gt.view(-1, video_len - 1, self.motion_dim)

        # get the initial state of LSTM
        state = self.trajGenerator.get_initial_state(batch_size)

        # generate each frame
        for idx in range(video_len - 1):
            # Scheduled sampling: use the ground truth or the generated last frame
            if random.random() > (1 - epsilon):
                cont_h = self.contEnc(images[:, :, idx])
            else:
                cont_h = self.contEnc(frames[-1])
            z_cont, _, _ = self.contSampler(cont_h, one_hot_matrix, use_mean=True)

            # Scheduled sampling: use the ground truth or the generated last difference map
            if idx > 0:
                if random.random() > (1 - epsilon):
                    motion_h_prev = motion_gt_mean[:, idx - 1]
                else:
                    current_diff = frames[-2] - frames[-1]
                    current_motion_h = self.motionEnc(current_diff)
                    motion_h_prev, _, _ = self.motionSampler(current_motion_h, one_hot_matrix, use_mean=True)
                _, state = self.trajGenerator.one_step(motion_h_prev, one_hot_matrix, state)

            if mode == 'random':
                z_motion = self.get_rand_var(batch_size, self.motion_dim)
            elif mode == 'sample':
                z_motion = z_motion_gt[:, idx]
            elif mode == 'mean':
                z_motion = motion_gt_mean[:, idx]
            else:
                raise ValueError('mode %s is not supported' % mode)
            motion_state, _ = self.trajGenerator.one_step(z_motion, one_hot_matrix, state)

            # combine the motion and content features
            comb_feat = self.combLayer(z_cont, motion_state.squeeze(), one_hot_matrix)
            frame = self.videoDec(comb_feat)
            frames.append(frame)

        video = torch.stack(frames, dim=2)
        return video, mean, logvar



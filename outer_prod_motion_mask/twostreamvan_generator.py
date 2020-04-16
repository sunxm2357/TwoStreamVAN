import sys
sys.path.insert(0, '..')
import torch
from torch import nn
from torch.autograd import Variable
from separable_convolution.SeparableConvolution import SeparableConvolution
from utils.util import one_hot
import numpy as np
import math
import pdb
import torch.nn.functional as F
import random


class ConvLstmCell(nn.Module):
    """A convolutional LSTM cell (https://arxiv.org/abs/1506.04214)."""

    def __init__(self, channels, kernel_size, forget_bias=1, activation=F.tanh, bias=True):
        """Constructor
        :param kernel_size: The kernel size of the convolutional layer
        :param channels: Controls the number of input/output features of cell
        :param forget_bias: The bias for the forget gate
        :param activation: The activation function to use in the gates
        :param bias: Whether to use a bias for the convolutional layer
        """
        super(ConvLstmCell, self).__init__()

        self.forget_bias = forget_bias
        self.activation = activation

        self.conv = nn.Conv2d(channels * 2, channels * 4, kernel_size, padding=(kernel_size - 1) / 2,
                              bias=bias)

    def forward(self, input, state=None):
        """Forward method
        :param input: The current input to the ConvLSTM [batch_size, seq_len, c, h, w]
        :param state: The previous state of the ConvLSTM (the concatenated memory cell and hidden state)
        """
        input_shape = input.shape
        if state is None:
            # initialize the state as zero
            state_shape = (input_shape[0], input_shape[2]*2, input_shape[3], input_shape[4])
            state = Variable(torch.ones(state_shape), requires_grad=False)
            if torch.cuda.is_available():
                state = state.cuda()
        seq_len = input_shape[1]
        c, h = torch.chunk(state, 2, dim=1)
        outputs = []
        for t in range(seq_len):
            current_input = input[:, t]
            conv_input = torch.cat((current_input, h), dim=1)
            conv_output = self.conv(conv_input)
            (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
            c = c * F.sigmoid(f + self.forget_bias) + F.sigmoid(i) * self.activation(j)
            h = self.activation(c) * F.sigmoid(o)
            outputs.append(h)
        state = torch.cat((c, h), dim=1)
        output = torch.stack(outputs, dim=1)
        return output, state


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
        :param gf_dim: The number of filters in the first layer
        """
        super(MotionEnc, self).__init__()
        self.gf_dim = gf_dim

        # resolution: 64 -> 32
        conv1_1 = nn.Conv2d(input_dim, gf_dim, 5, padding=2)
        bn1_1 = nn.BatchNorm2d(gf_dim)
        relu1_1 = nn.ReLU()

        conv1_2 = nn.Conv2d(gf_dim, gf_dim, 5, padding=2)
        bn1_2 = nn.BatchNorm2d(gf_dim)
        relu1_2 = nn.ReLU()

        pool1 = nn.MaxPool2d(2)

        # resolution: 32 -> 16
        conv2_1 = nn.Conv2d(gf_dim, gf_dim * 2, 5, padding=2)
        bn2_1 = nn.BatchNorm2d(gf_dim * 2)
        relu2_1 = nn.ReLU()

        conv2_2 = nn.Conv2d(gf_dim * 2, gf_dim * 2, 5, padding=2)
        bn2_2 = nn.BatchNorm2d(gf_dim * 2)
        relu2_2 = nn.ReLU()

        pool2 = nn.MaxPool2d(2)

        # resolution: 16 -> 8
        conv3_1 = nn.Conv2d(gf_dim * 2, gf_dim * 2, 7, padding=3)
        bn3_1 = nn.BatchNorm2d(gf_dim * 2)
        relu3_1 = nn.ReLU()

        conv3_2 = nn.Conv2d(gf_dim * 2, gf_dim, 7, padding=3)
        bn3_2 = nn.BatchNorm2d(gf_dim)
        relu3_2 = nn.ReLU()
        pool3 = nn.MaxPool2d(2)

        self.main = nn.Sequential(conv1_1, bn1_1, relu1_1,
                                  conv1_2, bn1_2, relu1_2, pool1,
                                  conv2_1, bn2_1, relu2_1,
                                  conv2_2, bn2_2, relu2_2, pool2,
                                  conv3_1, bn3_1, relu3_1,
                                  conv3_2, bn3_2, relu3_2, pool3)

        fc = nn.Linear(8 * 8 * gf_dim, 512)
        fc_bn = nn.BatchNorm1d(512)
        fc_relu = nn.ReLU()
        self.fc_block = nn.Sequential(fc, fc_bn, fc_relu)

    def fc_forward(self, feature):
        """
        :param feature: the hidden embedding of the difference map after CNN layers  [batch, gf_dim, 8, 8]
        :return: the hidden embedding of the difference map after the FC layer with the size [batch, 512]
        """
        feature = self.fc_block(feature.view(-1, 8 * 8 * self.gf_dim))
        return feature

    def forward(self, input_diff):
        """Forward method
        :param input_diff: A difference frame [batch_size, 1, h, w]
        :return: the hidden embedding of the difference map after CNN layers with the size [batch, gf_dim, 8, 8]
        """
        feature = self.main(input_diff)
        return feature


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


class TrajGenerator(nn.Module):
    """
        Adopting convLSTM to encode the sequence history conditioned on the action class
    """
    def __init__(self, latent_dim, num_categories, use_bn=False):
        """
        Recover the 2D feature map of the last diff map; encode the historical info via a convLSTM
        :param latent_dim: int, the dimension of the motion latent space
        :param num_categories: int, the number of the action categories
        :param use_bn: bool, specify if using the batch normalization
        """
        super(TrajGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories

        # fc layer is used to recover the 2D feature map of the last diff map
        fc = nn.Linear(latent_dim + num_categories, 8 * 8 * 16)
        relu = nn.ReLU()
        if use_bn:
            fc_bn = nn.BatchNorm1d(8 * 8 * 16)
            self.fc_block = nn.Sequential(fc, fc_bn, relu)
        else:
            self.fc_block = nn.Sequential(fc, relu)

        # get the convLstm
        self.main = ConvLstmCell(channels=16, kernel_size=3)

    def fc_forward(self, motion_h, one_hot_matrix):
        """
        Recover the 2D feature map of the last diff map
        :param motion_h: the sampled motion latent vector with the size [batch, latent_dim]
        :param one_hot_matrix: the one-hot vector with the size [batch, num_categories]
        :return: the 2D feature map of the last diff map with the size [batch, 16, 8, 8]
        """
        input_fc = torch.cat((motion_h, one_hot_matrix), dim=-1)
        motion_h = self.fc_block(input_fc).view(-1, 16, 8, 8)
        return motion_h

    def forward(self, motion_h, state_0=None):
        """
        :param motion_h: feature map encoded by the motionEnc or the random vector sampled
        :param state_0: the convLSTM state before this update
        :return output: the LSTM output with the size [batch, 16, 8, 8]
        :return state_1: the convLSTM state after this update
        """
        output, state_1 = self.main(motion_h, state_0)
        return output, state_1


class MCInput(nn.Module):
    """
    Compute the input feature map of the Content Generator and the Motion Generator
    """
    def __init__(self, content_dim, num_categories, gf_dim=16, joint=False, use_bn=False):
        """
        :param content_dim: int, the dimension of the content latent space
        :param num_categories: int, the number of action categories
        :param gf_dim: int, the base number of channels
        :param joint: bool, specify when the input to the motion encoder is
                      the outer-product of the content and motion vectors
        :param use_bn: bool, specify if using the batch normalization
        """
        super(MCInput, self).__init__()
        self.joint = joint
        self.gf_dim = gf_dim

        # FC layers for the content vector
        fc1 = nn.Linear(content_dim + num_categories, gf_dim * 32)
        fc_relu1 = nn.ReLU()

        fc2 = nn.Linear(gf_dim * 32, 8 * 8 * gf_dim)
        fc_relu2 = nn.ReLU()

        if use_bn:
            fc_bn1 = nn.BatchNorm1d(gf_dim * 32)
            fc_bn2 = nn.BatchNorm1d(8 * 8 * gf_dim)
            self.fc_cont1 = nn.Sequential(fc1, fc_bn1, fc_relu1)
            self.fc_cont2 = nn.Sequential(fc2, fc_bn2, fc_relu2)
        else:
            self.fc_cont1 = nn.Sequential(fc1, fc_relu1)
            self.fc_cont2 = nn.Sequential(fc2, fc_relu2)

        if joint:
            # FC layer for the motion vector
            fc_motion = nn.Linear(8 * 8 * 16, 64)
            relu_motion = nn.ReLU()
            if use_bn:
                bn_motion = nn.BatchNorm1d(64)
                self.fc_motion = nn.Sequential(fc_motion, bn_motion, relu_motion)
            else:
                self.fc_motion = nn.Sequential(fc_motion, relu_motion)

            # FC layer for the vector after combining
            fc_comb = nn.Linear((gf_dim * 32) * 64, 8 * 8 * 16)
            relu_comb = nn.ReLU()
            if use_bn:
                bn_comb = nn.BatchNorm1d(8 * 8 * 16)
                self.fc_comb = nn.Sequential(fc_comb, bn_comb, relu_comb)
            else:
                self.fc_comb = nn.Sequential(fc_comb, relu_comb)

    def cont_forward(self, cont_feat, one_hot_matrix):
        """
        Forward method for the content learning
        :param cont_feat: the content feature with the size [batch, cont_dim]
        :param one_hot_matrix: the one-hot vector with the size [batch, num_categories]
        :return: the input tensor to the content generator with the size [batch, gf_dim , 8, 8]
        """
        input_fc1 = torch.cat((cont_feat, one_hot_matrix), dim=1)
        tmp = self.fc_cont1(input_fc1)
        cont_state = self.fc_cont2(tmp).view(-1, self.gf_dim, 8, 8)
        return cont_state

    def forward(self, cont_feat, motion_feat, one_hot_matrix):
        """
        Forward method for the motion learning
        :param cont_feat: the content feature with the size [batch, cont_dim]
        :param motion_feat: the motion feature with the size [batch, 8 * 8 * 16]
        :param one_hot_matrix: the one-hot vector with the size [batch, num_categories]
        :return: the input tensor to the content generator with the size [batch, gf_dim , 8, 8]
                the input tensor to the content generator with the size [batch, 16 , 8, 8]
        """
        input_fc1 = torch.cat((cont_feat, one_hot_matrix), dim=1)
        tmp_cont = self.fc_cont1(input_fc1)
        cont_state = self.fc_cont2(tmp_cont).view(-1, self.gf_dim, 8, 8)
        if self.joint:
            tmp_motion = self.fc_motion(motion_feat.view(-1, 8 * 8 * 16))
            comb_feat = torch.bmm(tmp_cont.unsqueeze(2), tmp_motion.unsqueeze(1)).view(-1, (self.gf_dim * 32) * 64)
            motion_state = self.fc_comb(comb_feat).view(-1, 16, 8, 8)
        else:
            motion_state = motion_feat
        return cont_state, motion_state


class MotionGen(nn.Module):
    """
    Motion Generator: generate the adaptive kernels for multiple scales
    """
    def __init__(self, ac_kernel=5, kernel_layer=4):
        """
        :param ac_kernel: int, the kernel size of the adaptive convolution
        :param kernel_layer: int, the number of layers to apply adaptive convolutions
        """
        super(MotionGen, self).__init__()

        self.ac_kernel = ac_kernel
        self.kernel_layer = kernel_layer

        # resolution: 8 -> 8
        self.block2 = self.upsample_conv_block(16, 128, transpose_kernel=3, transpose_stride=1, transpose_pad=1)
        if self.kernel_layer >= 4:
            self.branch_block2, self.kernel_h_block2, self.kernel_v_block2, self.mask_block2 = self.mask_motion_block(32)

        # resolution: 8 -> 16
        self.block3 = self.upsample_conv_block(96, 64)
        if self.kernel_layer >= 3:
            self.branch_block3, self.kernel_h_block3, self.kernel_v_block3, self.mask_block3 = self.mask_motion_block(22)

        # resolution: 16 -> 32
        self.block4 = self.upsample_conv_block(42, 32)
        if self.kernel_layer >= 2:
            self.branch_block4, self.kernel_h_block4, self.kernel_v_block4, self.mask_block4 = self.mask_motion_block(16)

        # resolution: 32 -> 64
        self.block5 = self.upsample_conv_block(16, 16)
        if self.kernel_layer >= 1:
            self.branch_block5, self.kernel_h_block5, self.kernel_v_block5, self.mask_block5 = self.mask_motion_block(16)

    def upsample_conv_block(self, input_dim, output_dim, transpose_kernel=4, transpose_stride=2, transpose_pad=1):
        """
        The convolution block for upsampling
        :param input_dim: int, the dimension of the input of this convolution block
        :param output_dim: int, the dimension of the output of this comvolution block
        :param transpose_kernel: int, the kernel size of the transpose 2d convolution
        :param transpose_stride: int, the stride of the transpose 2d convolution
        :param transpose_pad: int, the padding of the transpose 2d convolution
        :return: the sequence model
        """
        deconv1 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=transpose_kernel, stride=transpose_stride, padding=transpose_pad)
        bn1 = nn.BatchNorm2d(output_dim)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(output_dim)
        relu2 = nn.ReLU()
        block = nn.Sequential(deconv1, bn1, relu1, conv2, bn2, relu2)
        return block

    def mask_motion_block(self, input_dim):
        """
        The branch to generate the motion kernels and mask at each scale
        :param input_dim: the input dimension of this branch
        :return: different sub-models for the horizontal, vertical kernels and the motion mask
        """

        # the convolution layer shared by the kernels and mask
        conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(16)
        relu1 = nn.ReLU()
        branch_block = nn.Sequential(conv1, bn1, relu1)

        # the cnn layer for the horizontal kernel
        conv_motion_h = nn.Conv2d(16, self.ac_kernel, kernel_size=3, padding=1)
        bn_motion_h = nn.BatchNorm2d(self.ac_kernel)
        relu_motion_h = nn.ReLU()
        motion_h = nn.Conv2d(self.ac_kernel, self.ac_kernel, kernel_size=3, padding=1)
        kernel_h_block = nn.Sequential(conv_motion_h, bn_motion_h, relu_motion_h, motion_h)

        # the cnn layer for the vertical kernel
        conv_motion_v = nn.Conv2d(16, self.ac_kernel, kernel_size=3, padding=1)
        bn_motion_v = nn.BatchNorm2d(self.ac_kernel)
        relu_motion_v = nn.ReLU()
        motion_v = nn.Conv2d(self.ac_kernel, self.ac_kernel, kernel_size=3, padding=1)
        kernel_v_block = nn.Sequential(conv_motion_v, bn_motion_v, relu_motion_v, motion_v)

        # the cnn layer for the motion mask
        conv1_mask = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        bn1_mask = nn.BatchNorm2d(16)
        relu1_mask = nn.ReLU()
        conv2_mask = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        tanh_mask = nn.Tanh()
        mask_block = nn.Sequential(conv1_mask, bn1_mask, relu1_mask, conv2_mask, tanh_mask)

        return branch_block, kernel_h_block, kernel_v_block, mask_block

    def forward(self, input):
        """
        Forward method
        :param input: variable with the size [batch, 16, 8, 8]
        :return: kernels and mask for all scales
        """

        # kernels and mask for 8x8 feature map
        feat8 = self.block2(input)
        if self.kernel_layer >= 4:
            branch_feat8 = self.branch_block2(feat8[:, :32])
            kernel_h_8, kernel_v_8, mask_8 = self.kernel_h_block2(branch_feat8), self.kernel_v_block2(branch_feat8), self.mask_block2(branch_feat8)
            mask_8 = (mask_8 + 1)/2
        else:
            kernel_h_8, kernel_v_8, mask_8 = None, None, None

        # kernels and mask for 16x16 feature map
        feat16 = self.block3(feat8[:, 32:])
        if self.kernel_layer >= 3:
            branch_feat16 = self.branch_block3(feat16[:, :22])
            kernel_h_16, kernel_v_16, mask_16 = self.kernel_h_block3(branch_feat16), self.kernel_v_block3(branch_feat16), self.mask_block3(branch_feat16)
            mask_16 = (mask_16 + 1)/2
        else:
            kernel_h_16, kernel_v_16, mask_16 = None, None, None

        # kernels and mask for 32x32 feature map
        feat32 = self.block4(feat16[:, 22:])
        if self.kernel_layer >= 2:
            branch_feat32 = self.branch_block4(feat32[:, :16])
            kernel_h_32, kernel_v_32, mask_32 = self.kernel_h_block4(branch_feat32), self.kernel_v_block4(branch_feat32), self.mask_block4(branch_feat32)
            mask_32 = (mask_32 + 1) / 2
        else:
            kernel_h_32, kernel_v_32, mask_32 = None, None, None

        # kernels and mask for 64x64 feature map
        feat64 = self.block5(feat32[:, 16:])
        if self.kernel_layer >= 1:
            branch_feat64 = self.branch_block5(feat64)
            kernel_h_64, kernel_v_64, mask_64 = self.kernel_h_block5(branch_feat64), self.kernel_v_block5(branch_feat64), self.mask_block5(branch_feat64)
            mask_64 = (mask_64 + 1) / 2
        else:
            kernel_h_64, kernel_v_64, mask_64 = None, None, None

        kernel_h_pyramid = [kernel_h_8, kernel_h_16, kernel_h_32, kernel_h_64]
        kernel_v_pyramid = [kernel_v_8, kernel_v_16, kernel_v_32, kernel_v_64]
        mask_pyramid = [mask_8, mask_16, mask_32, mask_64]

        return {'kernel_h': kernel_h_pyramid, 'kernel_v': kernel_v_pyramid, 'mask': mask_pyramid}


class ContGen(nn.Module):
    """
    Content Generator
    """
    def __init__(self, output_dim, gf_dim=16, no_mask=False, ac_kernel=5, use_bn=False, kernel_layer=4):
        """
        :param output_dim: int, the number of color channels of the output image
        :param gf_dim: int, the base number of channels
        :param no_mask: bool, specify when not using the motion mask
        :param ac_kernel: int, the size of the adaptive convolutional kernels
        :param use_bn: bool, specify if using the batch normalization
        :param kernel_layer: int, the number of layers to apply convolutions
        """
        super(ContGen, self).__init__()
        self.video_feat_dims = [gf_dim, gf_dim*8, gf_dim*2, 3]
        self.no_mask = no_mask
        self.ac_kernel = ac_kernel
        self.kernel_layer = kernel_layer

        # resolution: 8 -> 16
        deconv1 = nn.ConvTranspose2d(self.video_feat_dims[0], self.video_feat_dims[1], kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=False)
        relu1 = nn.ReLU()

        if use_bn:
            bn1 = nn.BatchNorm2d(self.video_feat_dims[1])
            self.conv1_block = nn.Sequential(deconv1, bn1, relu1)
        else:
            self.conv1_block = nn.Sequential(deconv1, relu1)

        # resolution: 16 -> 32
        deconv2 = nn.ConvTranspose2d(self.video_feat_dims[1], self.video_feat_dims[1]//2, kernel_size=3, stride=1, padding=1, bias=False)
        relu2 = nn.ReLU()
        deconv3 = nn.ConvTranspose2d(self.video_feat_dims[1]//2, self.video_feat_dims[2], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        relu3 = nn.ReLU()

        if use_bn:
            bn2 = nn.BatchNorm2d(self.video_feat_dims[1]/2)
            bn3 = nn.BatchNorm2d(self.video_feat_dims[2])
            self.conv2_block = nn.Sequential(deconv2, bn2, relu2, deconv3, bn3, relu3)
        else:
            self.conv2_block = nn.Sequential(deconv2, relu2, deconv3, relu3)

        # resolution: 32 -> 64
        deconv4 = nn.ConvTranspose2d(self.video_feat_dims[2], self.video_feat_dims[2], kernel_size=3, stride=1, padding=1, bias=False)
        relu4 = nn.ReLU()
        deconv5 = nn.ConvTranspose2d(self.video_feat_dims[2], self.video_feat_dims[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        relu5 = nn.ReLU()
        deconv6 = nn.ConvTranspose2d(self.video_feat_dims[0], output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        tanh = nn.Tanh()

        if use_bn:
            bn4 = nn.BatchNorm2d(self.video_feat_dims[2])
            bn5 = nn.BatchNorm2d(self.video_feat_dims[0])
            self.conv3_block = nn.Sequential(deconv4, bn4, relu4, deconv5, bn5, relu5, deconv6, tanh)
        else:
            self.conv3_block = nn.Sequential(deconv4, relu4,  deconv5, relu5, deconv6, tanh)

        # layers for the motion fusion
        self.modulePad = torch.nn.ReplicationPad2d([int(math.floor(self.ac_kernel / 2.0)), int(math.floor(self.ac_kernel / 2.0)),
                                                    int(math.floor(self.ac_kernel / 2.0)), int(math.floor(self.ac_kernel / 2.0))])

        self.separableConvolution = SeparableConvolution.apply

    def fusion(self, feat, kernel_v, kernel_h, mask):
        """
        single layer motion fusion
        :param feat: the content feature map
        :param kernel_v: the vertical kernels
        :param kernel_h: the horizontal kernels
        :param mask: the motion mask
        :return: the content feature map after fusing motion
        """
        tmp_feat = self.separableConvolution(self.modulePad(feat), kernel_v, kernel_h, self.ac_kernel)
        if self.no_mask:
            feat = tmp_feat
        else:
            num_feat = int(tmp_feat.shape[1])
            mask = mask.repeat(1, num_feat, 1, 1)
            feat = mask * tmp_feat + (1 - mask) * feat
        return feat

    def forward(self, cont_feat, transforms=None):
        """
        :param input1: content input [batch_size. gf_dim, 8, 8]
        :return: a tensor [batch_size, output_dim, h, w]
        """
        feat_8 = cont_feat
        scale_feats = []

        if (not transforms is None) and (self.kernel_layer >= 4):
            feat_8 = self.fusion(feat_8, transforms['kernel_v'][0], transforms['kernel_h'][0], transforms['mask'][0])
        if self.kernel_layer >= 4:
            scale_feats.append(feat_8)

        feat_16 = self.conv1_block(feat_8)
        if (not transforms is None) and (self.kernel_layer >= 3):
            feat_16 = self.fusion(feat_16, transforms['kernel_v'][1], transforms['kernel_h'][1], transforms['mask'][1])
        if self.kernel_layer >= 3:
            scale_feats.append(feat_16)

        feat_32 = self.conv2_block(feat_16)
        if (not transforms is None) and (self.kernel_layer >= 2):
            feat_32 = self.fusion(feat_32, transforms['kernel_v'][2], transforms['kernel_h'][2], transforms['mask'][2])
        if self.kernel_layer >= 2:
            scale_feats.append(feat_32)

        output = self.conv3_block(feat_32)
        if (not transforms is None) and (self.kernel_layer >= 1):
            output = self.fusion(output, transforms['kernel_v'][3], transforms['kernel_h'][3], transforms['mask'][3])

        return output, scale_feats


class Generator(nn.Module):
    """
    Generative part of the GAN model
    """
    def __init__(self, num_categories, n_channels, motion_dim=64, cont_dim=512, no_mask=False, joint=False,
                 gf_dim=16, ac_kernel=5, use_bn=False, kernel_layer=4):
        """
        :param num_categories: int, the number of action categories
        :param n_channels: int, the number of color channels of the image
        :param motion_dim: int, the dimension of the motion latent space
        :param cont_dim: int, the dimension of the content latent space
        :param no_mask: bool, specify when not using the motion mask
        :param joint: bool, specify when the input to the motion encoder is
                      the outer-product of the content and motion vectors
        :param gf_dim: int, the base number of channels
        :param ac_kernel: int, the kernel size of the adaptive convolution
        :param use_bn: bool, specify if using the batch normalization
        :param kernel_layer: int, the number of layers to apply adaptive convolutions
        """
        super(Generator, self).__init__()
        self.num_categories = num_categories
        self.cont_dim = cont_dim
        self.motion_dim = motion_dim
        self.joint = joint
        self.contEnc = ImgEnc(n_channels, gf_dim=gf_dim, use_bn=use_bn)
        self.contSampler = Sampler(gf_dim * 32 + num_categories, h_dim=cont_dim)
        self.motionEnc = MotionEnc(n_channels)
        self.motionSampler = Sampler(512 + num_categories, h_dim=motion_dim)
        self.trajGenerator = TrajGenerator(motion_dim, self.num_categories, use_bn=use_bn)
        self.contMotionStateGen = MCInput(cont_dim, num_categories, gf_dim=gf_dim, joint=joint, use_bn=use_bn)
        self.kernelGen = MotionGen(ac_kernel=ac_kernel, kernel_layer=kernel_layer)
        self.videoDec = ContGen(n_channels, gf_dim=gf_dim, no_mask=no_mask, ac_kernel=ac_kernel,
                                use_bn=use_bn, kernel_layer=kernel_layer)\

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

    @staticmethod
    def get_one_var(shape):
        m = torch.ones(shape)
        if torch.cuda.is_available():
            m = Variable(m.cuda())
        else:
            m = Variable(m)
        return m

    def reconstruct_one_frame(self, cat, input_frame=None, mode='sample'):
        """
        reconstruct the input frame
        :param input_frame: Variable of a tensor [batch_size, c, h, w]
        :param cat: Variable of a tensor [batch_size, num_categories]
        :param mode: sample or random for content latent space
        :return: reconstructed_frame, mean, logvar for content latent space
        """
        # prepare one_hot vector
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

        cont_state = self.contMotionStateGen.cont_forward(z_cont, one_hot_matrix)
        frame, scale_feat = self.videoDec(cont_state)
        return frame, mean, logvar, scale_feat

    def predict_next_frame(self, prev_frame, diff_frames, timestep, cat, mode='sample'):
        """
        predict the next frame given the prev_frame, diff_frames
        :param prev_frame: Variable of a tensor [batch_size, c, h, w]
        :param diff_frames: Variable of a tensor [batch_size, c, video_len-1, h, w]
        :param timestep: int which indicates which time step to predict
        :param cat: Variable of a tensor [batch_size, num_categories]
        :param mode: sample or random of motion latent space
        :return:
        """
        # prepare one_hot vector
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))
        batch_size, c, h, w = int(prev_frame.shape[0]), int(prev_frame.shape[1]), int(prev_frame.shape[2]), int(prev_frame.shape[3])

        #################################### Content encoding #######################################
        # get the content feature
        cont_h = self.contEnc(prev_frame)
        z_cont, _, _ = self.contSampler(cont_h, one_hot_matrix, use_mean=True)

        #################################### Motion encoding #######################################
        # get the motion feature for all diff maps
        motion_input = diff_frames[:, :, :timestep].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        motion_h = self.motionEnc(motion_input).view(-1, timestep, 16, 8, 8)

        # get the motion feature for the last diff map
        motion_h_cur = motion_h[:, -1]
        ## motion_h_cur: [batch, 512]
        motion_h_cur = self.motionEnc.fc_forward(motion_h_cur)

        # z_motion_cur expected shape: [batch, self.motion_dim]
        if mode == 'sample':
            z_motion_cur, mean, logvar = self.motionSampler(motion_h_cur, one_hot_matrix)
        elif mode == 'random':
            z_motion_cur = self.get_rand_var(batch_size, self.motion_dim)
            mean, logvar = None, None
        elif mode == 'mean':
            z_motion_cur, mean, logvar = self.motionSampler(motion_h_cur, one_hot_matrix, use_mean=True)
        else:
            raise ValueError('mode %s is not supported' % mode)

        # motion_h_cur: [batch_size, 1, 16, 8, 8]
        # motion_h is a sequence which is passed into convLSTM with the size [batch_size, timestep, 16, 8, 8]
        motion_h_cur = self.trajGenerator.fc_forward(z_motion_cur, one_hot_matrix).unsqueeze(1)
        if timestep > 1:
            motion_h_prev = motion_h[:, :-1]
            motion_h = torch.cat((motion_h_prev, motion_h_cur), dim=1)
        else:
            motion_h = motion_h_cur

        lstm_input = motion_h
        motion_states, _ = self.trajGenerator(lstm_input)
        # motion_state: [batch_size, seq_len, 16, 8, 8]
        motion_state = motion_states[:, -1]

        ################################### Generate the next frame #######################################
        cont_state, motion_state = self.contMotionStateGen(z_cont, motion_state, one_hot_matrix)
        transforms = self.kernelGen(motion_state)
        frame, scale_feat = self.videoDec(cont_state, transforms)
        return frame, mean, logvar, scale_feat

    def full_test(self, cat, video_len):
        """
        test mode to generate the entire video
        :param cat: variable, the action class with the size [batch]
        :param video_len: int, the desired length of the video
        :return:
        """
        masks = []
        frames = []
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))

        # generate the first image
        first_frame, _, _, _ = self.reconstruct_one_frame(cat, mode='random')
        frames.append(first_frame)

        # generate the last motion latent vectors for the generation at each time step
        batch_size = int(cat.shape[0])
        z_motion = self.get_rand_var(batch_size * (video_len - 1), self.motion_dim)
        one_hot_matrix_tile = one_hot_matrix.unsqueeze(1).repeat(1, video_len - 1, 1).view(-1, self.num_categories)
        motion_h_last = self.trajGenerator.fc_forward(z_motion, one_hot_matrix_tile).view(-1, video_len-1, 16, 8, 8)

        # get initial state
        state = None
        for idx in range(video_len-1):
            # get content encoding
            cont_h = self.contEnc(frames[-1])
            z_cont, _, _ = self.contSampler(cont_h, one_hot_matrix, use_mean=True)

            # get all previous motion encoding
            if idx > 0:
                current_diff = frames[-2] - frames[-1]
                motion_h_prev = self.motionEnc(current_diff)
                lstm_input = motion_h_prev.unsqueeze(1)
                _, state = self.trajGenerator(lstm_input, state_0=state)

            # update the convLSTM with the current motion variable
            lstm_input = motion_h_last[:, idx].unsqueeze(1)
            motion_state, _ = self.trajGenerator(lstm_input, state_0=state)

            # get the input for the content and motion generators
            cont_state, motion_state = self.contMotionStateGen(z_cont, motion_state.squeeze(), one_hot_matrix)

            # get the motion kernels and masks, generate the next frame
            transforms = self.kernelGen(motion_state)
            frame, _ = self.videoDec(cont_state, transforms)

            masks.append(transforms['mask'])
            frames.append(frame)

        video = torch.stack(frames, dim=1)
        return video, masks

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
        batch_size = int(cat.shape[0])
        video_len = int(diff_frames.shape[2]) + 1
        one_hot_matrix = Variable(one_hot(cat.data, self.num_categories))
        c, h, w = int(diff_frames.shape[1]), int(diff_frames.shape[3]), int(diff_frames.shape[4])


        ########################## reconstruct the first frame ##################################
        first_frame = images[:, :, 0]
        first_frame, _, _, _ = self.reconstruct_one_frame(cat, input_frame=first_frame, mode='mean')
        frames.append(first_frame)

        ########################### get motion embedding ##########################################
        # get the gt motion encoded vector without sampling from the generated distribution
        motion_input = diff_frames.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # motion_h expected shape [batch_size x (video_len - 1), gf_dim, 8, 8]
        motion_h = self.motionEnc(motion_input)
        # motion_h_fc expected shape [batch_size x (video_len - 1), 512]
        motion_h_fc = self.motionEnc.fc_forward(motion_h)

        # encode every timestep to the motion latent space
        one_hot_matrix_tile = one_hot_matrix.unsqueeze(1).repeat(1, video_len - 1, 1).view(-1, self.num_categories)
        if mode == 'random':
            z_motion = self.get_rand_var(batch_size * (video_len - 1), self.motion_dim)
            mean, logvar = None, None
        elif mode == 'sample':
            z_motion, mean, logvar = self.motionSampler(motion_h_fc, one_hot_matrix_tile)
        elif mode == 'mean':
            z_motion, mean, logvar = self.motionSampler(motion_h_fc, one_hot_matrix_tile, use_mean=True)
        else:
            raise ValueError('mode %s is not supported' % mode)

        # get 2D feature map from the latent vector
        motion_h_last = self.trajGenerator.fc_forward(z_motion, one_hot_matrix_tile).view(-1, video_len - 1, 16, 8, 8)

        ###################################### Generate the video frame by frame ################################
        motion_h = motion_h.view(-1, video_len - 1, 16, 8, 8)
        state = None
        scale_feats = []
        # generate each frame
        for idx in range(video_len-1):
            # schedule sampling for content encoder
            if random.random() > (1-epsilon):
                cont_h = self.contEnc(images[:, :, idx])
            else:
                cont_h = self.contEnc(frames[-1])
            # encode the content
            z_cont, _, _ = self.contSampler(cont_h, one_hot_matrix, use_mean=True)

            if idx > 0:
                # schedule sampling for motion encoder
                if random.random() > (1-epsilon):
                    motion_h_prev = motion_h[:, idx-1]
                else:
                    current_diff = frames[-2] - frames[-1]
                    motion_h_prev = self.motionEnc(current_diff)
                # input the previous diff map to update state
                lstm_input = motion_h_prev.unsqueeze(1)
                _, state = self.trajGenerator(lstm_input, state_0=state)
            # encode the motion
            # motion_h_cur expected shape [batch_size, 1, 16, 8, 8]
            motion_h_cur = motion_h_last[:, idx].unsqueeze(1)

            lstm_input = motion_h_cur
            motion_state, _ = self.trajGenerator(lstm_input, state_0=state)

            # decode the next frame
            cont_state, motion_state = self.contMotionStateGen(z_cont, motion_state.squeeze(), one_hot_matrix)
            transforms = self.kernelGen(motion_state)
            frame, scale_feat = self.videoDec(cont_state, transforms)
            frames.append(frame)

            # record the feature at each scale
            scale_feats += scale_feat

        video = torch.stack(frames, dim=2)
        return video, mean, logvar, scale_feats
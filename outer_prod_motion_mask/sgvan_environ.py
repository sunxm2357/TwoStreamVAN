import sys
sys.path.insert(0, '..')

from outer_prod_motion_mask.base_environ import BaseEnviron
from outer_prod_motion_mask.sgvan_generator import *
from outer_prod_motion_mask.discriminator import PatchImageDiscriminator, CategoricalVideoDiscriminator
from torch.autograd import Variable
import torch
import torch.optim as optim
from utils.util import makedir
from utils.scheduler import Scheduler
import numpy as np
import os
import cv2


if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class SGVAN(BaseEnviron):
    """
    The environment for SGVAN: training, testing, visualization, save/load checkpoints, etc
    """
    def __init__(self, gen_args, checkpoint_dir, log_dir, output_dir, video_len, action_set, actor_set, is_eval=True,
                 dis_args=None, loss_weights=None, pretrain_iters=0):
        """
        :param gen_args: dict, all the parameters/settings for the generator network
        :param checkpoint_dir:  str, the path to save/load checkpoints
        :param log_dir:  str, the path to save the log file
        :param output_dir: str, the path to save the generated videos
        :param video_len: str, the desired length of the generated videos
        :param action_set: list, the action set
        :param actor_set: list, the actor set
        :param is_eval: bool, specify for evaluation
        :param dis_args: dict, all the parameters/settings for the discriminator network
        :param loss_weights: dict, the weights for losses
        :param pretrain_iters: int, the number of iters for pretraining the content stream
        """

        super(SGVAN, self).__init__(gen_args, checkpoint_dir, log_dir, output_dir, video_len, action_set,
                                    actor_set, is_eval=is_eval)

        # define the generator and discriminator networks
        self.define_networks(gen_args, dis_args=dis_args)
        if torch.cuda.is_available():
            self.move_to_gpu()

        if not is_eval:
            # define the optimizers
            self.define_opts()

            # define the mechanism of the scheduled sampling
            self.schedule_sampler = Scheduler(1, 0, 200000+pretrain_iters, 400000+pretrain_iters, mode='linear')

            # define the loss weights
            self.motion_kl_weight = Scheduler(loss_weights['vid_m_kl_start'], loss_weights['vid_m_kl_end'], 200000+pretrain_iters,
                                              400000+pretrain_iters, mode='linear')
            self.c_kl = Scheduler(loss_weights['c_kl_start'], loss_weights['c_kl_end'], 0, 100000, mode='linear')
            self.img_m_kl = loss_weights['img_m_kl']
            self.c_img_dis = loss_weights['c_img_dis']

            self.xp_vs_xtilde = loss_weights['xp_vs_xtilde']

    def define_networks(self, gen_args, dis_args=None):
        """
        Define the architecture of networks
        :param gen_args: the args for the generator
        :param dis_args: the args for the discriminators
        """
        self.networks['generator'] = Generator(gen_args['num_categories'], gen_args['n_channels'],
                                               motion_dim=gen_args['motion_dim'], cont_dim=gen_args['cont_dim'],
                                               gf_dim=gen_args['gf_dim'])
        if not self.is_eval:
            self.networks['img_discriminator'] = PatchImageDiscriminator(dis_args['n_channels'])
            self.networks['vid_discriminator'] = CategoricalVideoDiscriminator(dis_args['n_channels'], dis_args['num_categories'])

    def define_opts(self):
        """
        Define Optimizers
        """
        # optimizer for content encoder
        contEnc_params = list(self.networks['generator'].contEnc.parameters()) + \
                         list(self.networks['generator'].contSampler.parameters())
        self.optimizers['optimize_contEnc'] = optim.Adam(contEnc_params, lr=0.0002, betas=(0.5, 0.999),
                                                         weight_decay=0.00001)

        # optimizer for motion encoder
        motionEnc_params = list(self.networks['generator'].motionEnc.parameters()) + \
                         list(self.networks['generator'].motionSampler.parameters())
        self.optimizers['optimize_motionEnc'] = optim.Adam(motionEnc_params, lr=0.0002, betas=(0.5, 0.999),
                                                           weight_decay=0.00001)

        # optimizer for the single decoder
        dec_params = list(self.networks['generator'].trajGenerator.parameters()) + \
                     list(self.networks['generator'].combLayer.parameters()) + \
                     list(self.networks['generator'].videoDec.parameters())
        self.optimizers['optimize_dec'] = optim.Adam(dec_params, lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        # optimizer for discriminators
        self.optimizers['optimize_d_img'] = optim.Adam(self.networks['img_discriminator'].parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optimizers['optimize_d_vid'] = optim.Adam(self.networks['vid_discriminator'].parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

    # ************************************************************************
    # ************************** Forward Methods *****************************
    # ************************************************************************

    def reconstruct_forward(self, ae_mode='sample', is_eval=False):
        """
        Forward method for the content stream
        :param ae_mode: the mode of the latent variable
                        'random': sample from the standard normal distribution N(0, 1)
                        'sample': sample from the posterior distribution q(z_c|x)
                        'mean': use the mean of q(z_c|x)
        :param is_eval: specify when validating or evaluating
        """
        self.task = 'recons'

        # select a random frame to reconstruct
        rand_idx = np.random.randint(low=0, high=self.video_len - 1)
        self.recons_img = self.images[:, :, rand_idx]

        if is_eval:
            torch.set_grad_enabled(False)

        # the ae pass
        self.recons_x_tilde, self.cont_mean, self.cont_logvar = \
            self.networks['generator'].reconstruct_one_frame(self.categories, self.recons_img, mode=ae_mode)

        # the gan pass
        self.recons_x_p, _, _ = self.networks['generator'].reconstruct_one_frame(self.categories, mode='random')

        if is_eval:
            torch.set_grad_enabled(True)

    def predict_forward(self, ae_mode='sample', is_eval=False):
        """
        Forward method for the easier task in the motion stream
        :param ae_mode: the mode of the latent variable
                        'random': sample from the standard normal distribution N(0, 1)
                        'sample': sample from the posterior distribution q(z_m|delta_x)
                        'mean': use the mean of q(z_m|delta_x)
        :param is_eval: specify when validating or evaluating
        """
        self.task = 'pred'

        # pick up predict timestep
        timestep = np.random.randint(low=1, high=self.video_len - 1)

        # prepare inputs
        self.prev_img = self.images[:, :, timestep-1]
        self.pred_target = self.images[:, :, timestep]
        diff_img = (self.images[:, :, : -1] - self.images[:, :, 1:])/2

        if is_eval:
            torch.set_grad_enabled(False)

        # the ae pass
        self.pred_x_tilde, self.motion_mean, self.motion_logvar = \
            self.networks['generator'].predict_next_frame(self.prev_img, diff_img, timestep, self.categories, mode=ae_mode)

        # the gan pass
        self.pred_x_p, _, _ = self.networks['generator'].predict_next_frame(self.prev_img, diff_img, timestep, self.categories, mode='random')

        if is_eval:
            torch.set_grad_enabled(True)

    def video_forward(self, eplison, ae_mode='sample',  is_eval=False):
        """
         Forward method for the easier task in the motion stream
         :param ae_mode: the mode of the latent variable
                         'random': sample from the standard normal distribution N(0, 1)
                         'sample': sample from the posterior distribution q(z_m|delta_x)
                         'mean': use the mean of q(z_m|delta_x)
         :param is_eval: specify when validating or evaluating
        """
        self.task = 'video_generate'

        # prepare inputs
        self.first_img = self.images[:, :, 0]
        diff_img = (self.images[:, :, : -1] - self.images[:, :, 1:]) / 2

        if is_eval:
            torch.set_grad_enabled(False)

        # the ae pass
        self.video_x_tilde, self.video_mean, self.video_logvar =\
            self.networks['generator'].reconstruct_seq(self.images, self.categories, diff_img, eplison, mode=ae_mode)

        # the gan pass
        self.video_x_p, _, _ = self.networks['generator'].reconstruct_seq(self.images, self.categories, diff_img, eplison, mode='random')

        if is_eval:
            torch.set_grad_enabled(True)

    # ************************************************************************
    # ************************** Loss  Functions *****************************
    # ************************************************************************

    def get_recons_pretrain_losses(self, c_kl):
        """
       VAE losses for pre-training the content stream
       :param c_kl: the weight of KL distance in the content training
       """
        ############################ loss for encoder ############################
        # kl divergence and reconstruction
        loss = {}
        loss['kld'] = self.kl_loss((self.cont_mean, self.cont_logvar))
        loss['recons_mse'] = self.mse_loss(self.recons_x_tilde, self.recons_img)
        loss['l_enc_recons'] = c_kl * loss['kld'] + 10000 * loss['recons_mse']
        self.losses['cont_enc'] = loss

        ########################### loss for decoder #############################
        # reconstruction loss
        loss = {}
        loss['recons_mse'] = self.mse_loss(self.recons_x_tilde, self.recons_img)
        loss['l_dec_recons'] = 10000 * loss['recons_mse']
        self.losses['dec'] = loss

    def get_recons_losses(self, c_kl):
        """
        VAE and GAN losses for the content learning
        :param c_kl: the weight of KL distance in the content training
        """
        ############################ loss for encoder ############################
        # kl divergence and reconstruction loss
        loss = {}
        loss['kld'] = self.kl_loss((self.cont_mean, self.cont_logvar))
        loss['recons_mse'] = self.mse_loss(self.recons_x_tilde, self.recons_img)
        loss['l_enc_recons'] = c_kl * loss['kld'] + 10000 * loss['recons_mse']
        self.losses['cont_enc'] = loss

        ########################### loss for decoder #############################
        # Reconstruction loss and discriminator loss
        loss = {}
        # Reconstruction loss
        loss['recons_mse'] = self.mse_loss(self.recons_x_tilde, self.recons_img)
        # Discriminator loss
        # # ae pass
        fake_score = self.networks['img_discriminator'](self.recons_x_tilde)
        all_ones = self.ones_like(fake_score.data)
        loss['recons_gan_x_tilde'] = self.gan_loss(fake_score, all_ones)
        # # gan pass
        fake_score = self.networks['img_discriminator'](self.recons_x_p)
        all_ones = self.ones_like(fake_score.data)
        loss['recons_gan_x_p'] = self.gan_loss(fake_score, all_ones)
        # total loss
        loss['l_dec_recons'] = 10000 * loss['recons_mse'] + \
                               self.c_img_dis * (loss['recons_gan_x_tilde'] + self.xp_vs_xtilde * loss['recons_gan_x_p'])
        self.losses['dec'] = loss

        ######################## loss for discriminator #########################
        loss = {}
        # # ae pass
        fake_score = self.networks['img_discriminator'](self.recons_x_tilde.detach())
        all_zeros = self.zeros_like(fake_score.data)
        loss['recons_gan_x_tilde'] = self.gan_loss(fake_score, all_zeros)
        # # gan pass
        fake_score = self.networks['img_discriminator'](self.recons_x_p.detach())
        all_zeros = self.zeros_like(fake_score.data)
        loss['recons_gan_x_p'] = self.gan_loss(fake_score, all_zeros)
        # # real pass
        real_score = self.networks['img_discriminator'](self.recons_img)
        all_ones = self.ones_like(real_score.data)
        loss['gan_x'] = self.gan_loss(real_score, all_ones)
        # total loss
        loss['l_dis_recons'] = loss['recons_gan_x_tilde'] + 0.01 * loss['recons_gan_x_p'] + loss['gan_x']

        self.losses['img_dis'] = loss

    def get_pred_losses(self):
        """
         VAE and GAN losses for the easier content learning
        """
        ################################# loss for encoder #######################################
        # kl divergence loss, L2 loss for video frames
        loss = {}
        loss['kld'] = self.kl_loss((self.motion_mean, self.motion_logvar))
        loss['pred_mse'] = self.mse_loss(self.pred_x_tilde, self.pred_target)
        loss['l_enc_pred'] = self.img_m_kl * loss['kld'] + 10000 * loss['pred_mse']
        self.losses['motion_enc'] = loss

        ################################# loss for decoder #####################################
        # L2 loss for video frames and the modified content feature map, discriminator loss
        loss = {}
        loss['pred_mse'] = self.mse_loss(self.pred_x_tilde, self.pred_target)

        # # ae pass
        fake_score = self.networks['img_discriminator'](self.pred_x_tilde)
        all_ones = self.ones_like(fake_score.data)
        loss['pred_gan_x_tilde'] = self.gan_loss(fake_score, all_ones)

        # # gan pass
        fake_score = self.networks['img_discriminator'](self.pred_x_p)
        all_ones = self.ones_like(fake_score.data)
        loss['pred_gan_x_p'] = self.gan_loss(fake_score, all_ones)

        loss['l_dec_pred'] = 10000 * loss['pred_mse'] + 10 * (loss['pred_gan_x_tilde'] + loss['pred_gan_x_p'])
        self.losses['dec'] = loss

        ############################### loss for discriminator ################################
        loss = {}
        # # ae pass
        fake_score = self.networks['img_discriminator'](self.pred_x_tilde.detach())
        all_zeros = self.zeros_like(fake_score.data)
        loss['pred_gan_x_tilde'] = self.gan_loss(fake_score, all_zeros)
        # # GAN pass
        fake_score = self.networks['img_discriminator'](self.pred_x_p.detach())
        all_zeros = self.zeros_like(fake_score.data)
        loss['pred_gan_x_p'] = self.gan_loss(fake_score, all_zeros)
        # # real data pass
        real_score = self.networks['img_discriminator'](self.pred_target)
        all_ones = self.ones_like(real_score.data)
        loss['gan_x'] = self.gan_loss(real_score, all_ones)

        loss['l_dis_pred'] = loss['pred_gan_x_tilde'] + loss['pred_gan_x_p'] + loss['gan_x']

        self.losses['img_dis'] = loss

    def get_video_loss(self, m_kl):
        """
        VAE and GAN losses for the harder motion learning
        :param m_kl: the weight of KL distance in the motion training
        """
        ############################### loss for encoder ######################################
        # kl divergence loss, L2 loss for video frames
        loss = {}
        loss['kld'] = self.kl_loss((self.video_mean, self.video_logvar))
        loss['pred_mse'] = self.mse_loss(self.video_x_tilde, self.images)
        loss['l_enc'] = m_kl * loss['kld'] + 10000 * loss['pred_mse']
        self.losses['vid_motion_enc'] = loss

        ################################# loss for decoder ####################################
        # L2 loss for video frames, discriminator loss
        loss = {}
        loss['pred_mse'] = self.mse_loss(self.video_x_tilde, self.images)

        # # ae pass
        fake_score, fake_categories = self.networks['vid_discriminator'](self.video_x_tilde)
        all_ones = self.ones_like(fake_score.data)
        loss['vid_gan_x_tilde'] = self.gan_loss(fake_score, all_ones)
        loss['vid_cat_x_tilde'] = self.category_criterion(fake_categories, self.categories)

        # # GAN pass
        fake_score, fake_categories = self.networks['vid_discriminator'](self.video_x_p)
        all_ones = self.ones_like(fake_score.data)
        loss['vid_gan_x_p'] = self.gan_loss(fake_score, all_ones)
        loss['vid_cat_x_p'] = self.category_criterion(fake_categories, self.categories)

        loss['l_dec'] = 10000 * loss['pred_mse'] + 1 * (loss['vid_gan_x_tilde'] + loss['vid_gan_x_p']) + \
                        1 * (loss['vid_cat_x_tilde'] + loss['vid_cat_x_p'])
        self.losses['vid_dec'] = loss

        ############################################ loss for discriminator ####################################
        loss = {}
        # # ae pass
        fake_score, fake_categories = self.networks['vid_discriminator'](self.video_x_tilde.detach())
        all_zeros = self.zeros_like(fake_score.data)
        loss['vid_gan_x_tilde'] = self.gan_loss(fake_score, all_zeros)
        # # GAN pass
        fake_score, fake_categories = self.networks['vid_discriminator'](self.video_x_p.detach())
        all_zeros = self.zeros_like(fake_score.data)
        loss['vid_gan_x_p'] = self.gan_loss(fake_score, all_zeros)
        # # real data
        real_score, real_categories = self.networks['vid_discriminator'](self.images)
        all_ones = self.ones_like(real_score.data)
        loss['vid_gan_x'] = self.gan_loss(real_score, all_ones)
        loss['vid_cat_x'] = self.category_criterion(real_categories, self.categories)

        loss['l_dis'] = loss['vid_gan_x_tilde'] + loss['vid_gan_x_p'] + loss['vid_gan_x'] + loss['vid_cat_x']
        self.losses['vid_dis'] = loss

    # ************************************************************************
    # ************************** Backward Methods ****************************
    # ************************************************************************

    def backward_dec(self):
        """
         Backward method for the general generator
         """
        self.optimizers['optimize_dec'].zero_grad()
        if self.task == 'pred':
            self.losses['dec']['l_dec_pred'].backward()
        elif self.task == 'recons':
            self.losses['dec']['l_dec_recons'].backward()
        elif self.task == 'video_generate':
            self.losses['vid_dec']['l_dec'].backward()
        else:
            raise ValueError('task %s is not supported' % self.task)
        self.optimizers['optimize_dec'].step()

    # ************************************************************************
    # ************************** Optimization Framework **********************
    # ************************************************************************

    def optimize_pred_parameters(self):
        """
        Optimization framework for the easier motion learning
        """
        self.freeze_cont_stream()

        # forward
        self.predict_forward()

        # get loss
        self.get_pred_losses()
        real_cost = np.mean([self.losses['img_dis']['gan_x'].data])
        fake_cost = np.mean([self.losses['img_dis']['pred_gan_x_tilde'].data, self.losses['img_dis']['pred_gan_x_p'].data])

        # backward
        equilibrium = 0.68
        margin = 0.4
        dec_update = True
        dis_update = True
        if margin is not None:
            if real_cost < equilibrium - margin or fake_cost < equilibrium - margin:
                dis_update = False
            if real_cost > equilibrium + margin or fake_cost > equilibrium + margin:
                dec_update = False
            if not (dec_update or dis_update):
                dec_update = True
                dis_update = True
        self.backward_motion_enc(retain=dec_update)
        if dec_update:
            self.backward_dec()
        if dis_update:
            self.backward_img_dis()
        self.free_cont_stream()

    def optimize_recons_pretrain_parameters(self, current_iter):
        """
        Optimization framework for pre-training content stream
        :param current_iter: the current number of iteration
        """
        # forward
        self.reconstruct_forward()
        # get loss
        c_kl = self.c_kl.get_value(current_iter)
        self.get_recons_pretrain_losses(c_kl)
        # backward
        self.backward_content_enc(retain=True)
        self.backward_dec()

    def optimize_recons_parameters(self, current_iter):
        """
        Optimization framework for the content learning
        :param current_iter: the current number of iteration
        """
        # forward
        self.reconstruct_forward()

        # get losses
        c_kl = self.c_kl.get_value(current_iter)
        self.get_recons_losses(c_kl)
        real_cost = np.mean([self.losses['img_dis']['gan_x'].data])
        fake_cost = np.mean([self.losses['img_dis']['recons_gan_x_tilde'].data, self.losses['img_dis']['recons_gan_x_p'].data])

        # backward
        equilibrium = 0.68
        margin = 0.5
        dec_update = True
        dis_update = True
        if margin is not None:
            if real_cost < equilibrium - margin or fake_cost < equilibrium - margin:
                dis_update = False
            if real_cost > equilibrium + margin or fake_cost > equilibrium + margin:
                dec_update = False
            if not (dec_update or dis_update):
                dec_update = True
                dis_update = True
        self.backward_content_enc(retain=dec_update)
        if dec_update:
            self.backward_dec()
        if dis_update:
            self.backward_img_dis()

    def optimize_vid_parameters(self, current_iter):
        """
        Optimization framework for the harder motion learning
        :param current_iter: the current number of iteration
        """
        self.freeze_cont_stream()

        # forward
        eplison = self.schedule_sampler.get_value(current_iter)
        self.video_forward(eplison, ae_mode='sample')

        # get losses
        motion_kld_weight = self.motion_kl_weight.get_value(current_iter)
        self.get_video_loss(motion_kld_weight)
        real_cost = np.mean([self.losses['vid_dis']['vid_gan_x'].data])
        fake_cost = np.mean([self.losses['vid_dis']['vid_gan_x_tilde'].data, self.losses['vid_dis']['vid_gan_x_p'].data])

        # backward
        equilibrium = 0.68
        margin = 0.5
        dec_update = True
        dis_update = True
        if margin is not None:
            if real_cost < equilibrium - margin or fake_cost < equilibrium - margin:
                dis_update = False
            if real_cost > equilibrium + margin or fake_cost > equilibrium + margin:
                dec_update = False
            if not (dec_update or dis_update):
                dec_update = True
                dis_update = True
        self.backward_motion_enc(retain=dec_update)
        if dec_update:
            self.backward_dec()
        if dis_update:
            self.backward_vid_dis()
        self.free_cont_stream()

    # ************************************************************************
    # ******************************** Test **********************************
    # ************************************************************************

    def full_test(self, cls_id, batch_size, video_len, current_iter, var_name, start_idx=0, is_eval=False, rm_npy=False):
        """
        :param cls_id:  int, the action index at test
        :param batch_size: int
        :param video_len: int, the desired length of the video
        :param current_iter: int, the current iteration so far
        :param var_name: str, the variable name for saving or tensorboard visualizing
        :param start_idx: int, the start index of the current batch
        :param is_eval: bool, specify when evaluating
        :param rm_npy: bool, specify to remove all npy files in the output folder
        :return: output_dir: str, the output path
        """
        # create the category matrix for the test class
        cat = cls_id * np.ones((batch_size,)).astype('int')
        if torch.cuda.is_available():
            self.categories = Variable(torch.from_numpy(cat)).cuda()
        else:
            self.categories = Variable(torch.from_numpy(cat))

        # generate the video with size [batch_size, video_len, c, h, w]
        video = self.networks['generator'].full_test(self.categories, video_len+2)
        # heat up the generator for two steps
        video = video[:, 2:]

        # create the output directory
        if is_eval:
            output_dir = os.path.join(self.output_dir, 'evaluation', str(current_iter))
        else:
            output_dir = os.path.join(self.output_dir, 'validation', str(current_iter))
        makedir(output_dir)

        # remove the existing npy files
        if rm_npy:
            os.system('rm %s' % os.path.join(output_dir, '*.npy'))

        # save original output to npy file
        # video_np [batch_size, c, video_len, h, w]
        video_np = video.cpu().data.numpy()
        self.save_video(video_np, output_dir, self.categories, start_idx=start_idx)
        if not is_eval:
            # save to tensorboard
            self.writer.add_video(var_name, (video.permute(0, 2, 1, 3, 4) + 1)/2, current_iter)
        return output_dir

    def freeze_cont_stream(self):
        """
        freeze the content params during the motion learning
        """
        for param in self.networks['generator'].contEnc.parameters():
            param.requires_grad = False
        for param in self.networks['generator'].contSampler.parameters():
            param.requires_grad = False

    def free_cont_stream(self):
        """
        free the content params after the motion learning
        """
        for param in self.networks['generator'].contEnc.parameters():
            param.requires_grad = True
        for param in self.networks['generator'].contSampler.parameters():
            param.requires_grad = True

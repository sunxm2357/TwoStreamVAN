import sys
sys.path.insert(0, '..')
import os
import cv2

import torch.optim as optim

from outer_prod_motion_mask.base_environ import BaseEnviron
from outer_prod_motion_mask.twostreamvan_generator import *
from outer_prod_motion_mask.discriminator import PatchImageDiscriminator, CategoricalVideoDiscriminator

from utils.util import  makedir
from utils.scheduler import Scheduler


if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class TwoStreamVAN(BaseEnviron):
    """
    The environment for TwoStreamVAN: training, testing, visualization, save/load checkpoints, etc
    Variants of TwoStreamVAN (TwoStreamVAN(-C), TwoStreamVAN(-M))
    """
    def __init__(self, gen_args, checkpoint_dir, log_dir, output_dir, video_len, action_set, actor_set, is_eval=False,
                 dis_args=None, loss_weights=None, pretrain_iters=0):
        """
        :param gen_args: dict, all the parameters/settings for the generator network
        :param checkpoint_dir: str, the path to save/load checkpoints
        :param log_dir: str, the path to save the log file
        :param output_dir: str, the path to save the generated videos
        :param video_len: str, the desired length of the generated videos
        :param action_set: list, the action set
        :param actor_set: list, the actor set
        :param is_eval: bool, specify for evaluation
        :param dis_args: dict, all the parameters/settings for the discriminator network
        :param loss_weights: dict, the weights for losses
        :param pretrain_iters: int, the number of iters for pretraining the content stream
        """
        super(TwoStreamVAN, self).__init__(gen_args, checkpoint_dir, log_dir, output_dir, video_len, action_set,
                                           actor_set, is_eval=is_eval)

        self.joint = gen_args['joint']
        self.layers = gen_args['kernel_layer']

        # define the generator and discriminator networks
        self.define_networks(gen_args, dis_args=dis_args)
        if torch.cuda.is_available():
            self.move_to_gpu()

        if not is_eval:
            # define the optimizers
            self.define_opts()

            # define the mechanism of the scheduled sampling
            self.schedule_sampler = Scheduler(1, 0, 150000 + pretrain_iters, 400000 + pretrain_iters, mode='linear')

            # define the loss weights
            self.motion_kl_weight = Scheduler(loss_weights['vid_m_kl_start'], loss_weights['vid_m_kl_end'], 150000 + pretrain_iters,
                                              400000 + pretrain_iters, mode='linear')
            # scheduler for c_kl is only activated in pre-training
            self.c_kl = Scheduler(loss_weights['c_kl_start'], loss_weights['c_kl_end'], 0, 100000, mode='linear')
            self.pred_scale_feat = loss_weights['pred_scale_feat']
            self.video_scale_feat = loss_weights['video_scale_feat']
            self.img_m_kl = loss_weights['img_m_kl']
            self.c_img_dis = loss_weights['c_img_dis']
            # TODO: add xp_vs_xtilde to the loss weights, 1 as default, 0.01 for syn-action
            self.xp_vs_xtilde = loss_weights['xp_vs_xtilde']

    def define_networks(self, gen_args, dis_args=None):
        """
        Define the architecture of networks
        :param gen_args: the args for the generator
        :param dis_args: the args for the discriminators
        """
        self.networks['generator'] = Generator(gen_args['num_categories'], gen_args['n_channels'],
                                               motion_dim=gen_args['motion_dim'], cont_dim=gen_args['cont_dim'],
                                               no_mask=gen_args['no_mask'], joint=gen_args['joint'],
                                               ac_kernel=gen_args['ac_kernel'], gf_dim=gen_args['gf_dim'],
                                               use_bn=gen_args['use_bn'], kernel_layer=gen_args['kernel_layer'])

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
        motionDec_params = list(self.networks['generator'].trajGenerator.parameters()) + \
                           list(self.networks['generator'].kernelGen.parameters())

        # optimizer for motion generator
        if self.joint:
            motionDec_params += list(self.networks['generator'].contMotionStateGen.fc_motion.parameters()) + \
                                list(self.networks['generator'].contMotionStateGen.fc_comb.parameters())
        self.optimizers['optimize_motionDec'] = optim.Adam(motionDec_params, lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        contDec_params = list(self.networks['generator'].videoDec.parameters()) + \
                         list(self.networks['generator'].contMotionStateGen.fc_cont1.parameters()) + \
                         list(self.networks['generator'].contMotionStateGen.fc_cont2.parameters())

        # optimizer for content generator
        self.optimizers['optimize_contDec'] = optim.Adam(contDec_params, lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        # optimizer for discriminators
        self.optimizers['optimize_d_img'] = optim.Adam(self.networks['img_discriminator'].parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optimizers['optimize_d_vid'] = optim.Adam(self.networks['vid_discriminator'].parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

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
        self.recons_x_tilde, self.cont_mean, self.cont_logvar, _ = \
            self.networks['generator'].reconstruct_one_frame(self.categories, self.recons_img, mode=ae_mode)

        # the gan pass
        self.recons_x_p, _, _, _ = self.networks['generator'].reconstruct_one_frame(self.categories, mode='random')

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
        self.pred_x_tilde, self.motion_mean, self.motion_logvar, self.motion_scale_feats = \
            self.networks['generator'].predict_next_frame(self.prev_img, diff_img, timestep, self.categories, mode=ae_mode)

        # get the ground truth of self.scale_feats
        with torch.set_grad_enabled(False):
            _, _, _, self.pred_scale_feats_gt = self.networks['generator'].reconstruct_one_frame(self.categories, self.pred_target, mode='mean')

        # the gan pass
        self.pred_x_p, _, _, _ = self.networks['generator'].predict_next_frame(self.prev_img, diff_img, timestep, self.categories, mode='random')

        if is_eval:
            torch.set_grad_enabled(True)

    def video_forward(self, eplison, ae_mode='sample', is_eval=False):
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
        self.video_x_tilde, self.video_mean, self.video_logvar, self.video_scale_feats =\
            self.networks['generator'].reconstruct_seq(self.images, self.categories, diff_img, eplison, mode=ae_mode)

        # the gan pass
        self.video_x_p, _, _, _ = self.networks['generator'].reconstruct_seq(self.images, self.categories, diff_img,
                                                                             eplison, mode='random')

        # get the ground truth of self.scale_feats
        with torch.set_grad_enabled(False):
            video_len = self.images.shape[2]
            self.video_scale_feats_gt = []
            for idx in range(video_len-1):
                _, _, _,  scale_feat_gt = self.networks['generator'].reconstruct_one_frame(self.categories, self.images[:, :, idx+1],
                                                                                          mode='mean')
                self.video_scale_feats_gt += scale_feat_gt

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
        self.losses['cont_dec'] = loss

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
        # total loss
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
        self.losses['cont_dec'] = loss

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
        # # real data
        real_score = self.networks['img_discriminator'](self.recons_img)
        all_ones = self.ones_like(real_score.data)
        loss['gan_x'] = self.gan_loss(real_score, all_ones)
        # total loss
        loss['l_dis_recons'] = loss['recons_gan_x_tilde'] + self.xp_vs_xtilde * loss['recons_gan_x_p'] + loss['gan_x']
        self.losses['img_dis'] = loss

    def get_pred_losses(self):
        """
         VAE and GAN losses for the easier content learning
        """
        ################################# loss for encoder #######################################
        # kl divergence loss, L2 loss for video frames and the modified content feature map
        loss = {}
        loss['kld'] = self.kl_loss((self.motion_mean, self.motion_logvar))
        loss['pred_mse'] = self.mse_loss(self.pred_x_tilde, self.pred_target)

        loss['scale_feat_loss'] = 0
        for (scale_feat, scale_feat_gt) in zip(self.motion_scale_feats, self.pred_scale_feats_gt):
            loss['scale_feat_loss'] += self.mse_loss(scale_feat, scale_feat_gt.detach())

        loss['l_enc_pred'] = self.img_m_kl * loss['kld'] + 10000 * loss['pred_mse'] + self.pred_scale_feat * loss['scale_feat_loss']
        self.losses['motion_enc'] = loss

        ################################# loss for decoder #####################################
        # L2 loss for video frames and the modified content feature map, discriminator loss
        loss = {}
        loss['pred_mse'] = self.mse_loss(self.pred_x_tilde, self.pred_target)

        loss['scale_feat_loss'] = 0
        for (scale_feat, scale_feat_gt) in zip(self.motion_scale_feats, self.pred_scale_feats_gt):
            loss['scale_feat_loss'] += self.mse_loss(scale_feat, scale_feat_gt.detach())

        # # ae pass
        fake_score = self.networks['img_discriminator'](self.pred_x_tilde)
        all_ones = self.ones_like(fake_score.data)
        loss['pred_gan_x_tilde'] = self.gan_loss(fake_score, all_ones)

        # # gan pass
        fake_score = self.networks['img_discriminator'](self.pred_x_p)
        all_ones = self.ones_like(fake_score.data)
        loss['pred_gan_x_p'] = self.gan_loss(fake_score, all_ones)

        loss['l_dec_pred'] = 10000 * loss['pred_mse'] + 10 * (loss['pred_gan_x_tilde'] + loss['pred_gan_x_p']) + \
                                 self.pred_scale_feat * loss['scale_feat_loss']
        self.losses['motion_dec'] = loss

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
        # kl divergence loss, L2 loss for video frames and the modified content feature map
        loss = {}
        loss['kld'] = self.kl_loss((self.video_mean, self.video_logvar))
        loss['pred_mse'] = self.mse_loss(self.video_x_tilde, self.images)

        loss['scale_feat_loss'] = 0
        for (scale_feat, scale_feat_gt) in zip(self.video_scale_feats, self.video_scale_feats_gt):
            loss['scale_feat_loss'] += self.mse_loss(scale_feat, scale_feat_gt)

        # make the scale feat loss independent of the video length
        video_len = self.images.shape[2]
        loss['scale_feat_loss'] /= (video_len - 1)
        loss['l_enc'] = m_kl * loss['kld'] + 10000 * loss['pred_mse'] + self.video_scale_feat * loss['scale_feat_loss']

        self.losses['vid_motion_enc'] = loss

        ################################# loss for decoder ####################################
        # L2 loss for video frames and the modified content feature map, discriminator loss
        loss = {}
        loss['pred_mse'] = self.mse_loss(self.video_x_tilde, self.images)

        loss['scale_feat_loss'] = 0
        for (scale_feat, scale_feat_gt) in zip(self.video_scale_feats, self.video_scale_feats_gt):
            loss['scale_feat_loss'] += self.mse_loss(scale_feat, scale_feat_gt)
        video_len = self.images.shape[2]
        loss['scale_feat_loss'] /= (video_len - 1)

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
                        loss['vid_cat_x_tilde'] + loss['vid_cat_x_p'] + self.video_scale_feat * loss['scale_feat_loss']
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

    def backward_motion_dec(self):
        """
        Backward method for the motion generator
        """
        self.optimizers['optimize_motionDec'].zero_grad()
        if self.task == 'pred':
            self.losses['motion_dec']['l_dec_pred'].backward()
        elif self.task == 'video_generate':
            self.losses['vid_dec']['l_dec'].backward()
        else:
            raise ValueError('task %s is not supported' % self.task)
        self.optimizers['optimize_motionDec'].step()

    def backward_cont_dec(self):
        """
        Backward method for the content generator
        """
        self.optimizers['optimize_contDec'].zero_grad()
        self.losses['cont_dec']['l_dec_recons'].backward()
        self.optimizers['optimize_contDec'].step()

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
            self.backward_motion_dec()
        else:
            self.del_motion_dec()
        if dis_update:
            self.backward_img_dis()
        else:
            self.del_img_dis()

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
        self.backward_cont_dec()

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
        fake_cost = np.mean([2*(1-self.xp_vs_xtilde) * self.losses['img_dis']['recons_gan_x_tilde'].data, 2*self.xp_vs_xtilde*self.losses['img_dis']['recons_gan_x_p'].data])

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
            self.backward_cont_dec()
        else:
            self.del_cont_dec()
        if dis_update:
            self.backward_img_dis()
        else:
            self.del_img_dis()

    def optimize_vid_parameters(self, current_iter):
        """
        Optimization framework for the harder motion learning
        :param current_iter: the current number of iteration
        """
        self.freeze_cont_stream()
        # forward
        epsilon = self.schedule_sampler.get_value(current_iter)
        self.video_forward(epsilon, ae_mode='sample')

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
            self.backward_motion_dec()
        else:
            self.del_motion_dec()
        if dis_update:
            self.backward_vid_dis()
        else:
            self.del_vid_dis()
        self.free_cont_stream()

    # ************************************************************************
    # ******************************** Test **********************************
    # ************************************************************************

    def full_test(self, cls_id, batch_size, video_len, current_iter, var_name, start_idx=0, is_eval=False, rm_npy=False,
                  get_seq=False, get_mask=False):
        """
        :param cls_id:  int, the action index at test
        :param batch_size: int
        :param video_len: int, the desired length of the video
        :param current_iter: int, the current iteration so far
        :param var_name: str, the variable name for saving or tensorboard visualizing
        :param start_idx: int, the start index of the current batch
        :param is_eval: bool, specify when evaluating
        :param rm_npy: bool, specify to remove all npy files in the output folder
        :param get_seq: bool, specify to save the video sequence
        :param get_mask: bool, specify to visualize the mask
        :return: output_dir: str, the output path
        """
        # create the category matrix for the test class
        cat = cls_id * np.ones((batch_size,)).astype('int')
        if torch.cuda.is_available():
            self.categories = Variable(torch.from_numpy(cat)).cuda()
        else:
            self.categories = Variable(torch.from_numpy(cat))

        # generate the video with size [batch_size, video_len, c, h, w]
        torch.set_grad_enabled(False)
        video, masks = self.networks['generator'].full_test(self.categories, video_len+2)
        torch.set_grad_enabled(True)
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
        # video_np [batch_size, video_len, c,  h, w]
        video_np = video.cpu().data.numpy().clip(-1, 1)
        self.save_video(video_np, output_dir, self.categories, start_idx=start_idx)

        # saving to tensorboard during the validation
        if not is_eval:
            # save to tensorboard
            # [batch_size, video_len, c, h, w]
            video = torch.clamp((video.permute(0, 2, 1, 3, 4) + 1)/2, 0, 1)
            self.writer.add_video(var_name, video, current_iter)

        # save the video sequences to the output folder
        if get_seq:
            video_seqs = ((video_np.transpose(0, 1, 3, 4, 2) + 1)/2 * 255).astype('uint8')
            video_seqs = np.concatenate(np.split(video_seqs, video_len, axis=1), axis=3).squeeze()

            img_dir = os.path.join(output_dir, 'imgs')
            makedir(img_dir)
            for v_idx, seq in enumerate(video_seqs):
                filename = os.path.join(img_dir, '%s_%03d.png' % (var_name, start_idx + v_idx))
                cv2.imwrite(filename, seq[:, :, ::-1])

        # save masks to the output folder
        if get_mask:
            mask_8 = []
            mask_16 = []
            mask_32 = []
            mask_64 = []
            for frame_mask in masks:
                if self.layers >= 4:
                    mask_8.append(frame_mask[0].cpu().numpy().squeeze().clip(0, 1))
                if self.layers >= 3:
                    mask_16.append(frame_mask[1].cpu().numpy().squeeze().clip(0, 1))
                if self.layers >= 2:
                    mask_32.append(frame_mask[2].cpu().numpy().squeeze().clip(0, 1))
                if self.layers >= 1:
                    mask_64.append(frame_mask[3].cpu().numpy().squeeze().clip(0, 1))
            if self.layers >= 4:
                mask_8 = np.concatenate(mask_8[2:], axis=2)
            if self.layers >= 3:
                mask_16 = np.concatenate(mask_16[2:], axis=2)
            if self.layers >= 2:
                mask_32 = np.concatenate(mask_32[2:], axis=2)
            if self.layers >= 1:
                mask_64 = np.concatenate(mask_64[2:], axis=2)
            mask_dir = os.path.join(output_dir, 'masks')
            makedir(mask_dir)
            for v_idx in range(batch_size):
                if self.layers >= 4:
                    filename = os.path.join(mask_dir, '%s_%03d_mask_8.png' % (var_name, start_idx + v_idx))
                    cv2.imwrite(filename, (mask_8[v_idx] * 255).astype('uint8'))
                if self.layers >= 3:
                    filename = os.path.join(mask_dir, '%s_%03d_mask_16.png' % (var_name, start_idx + v_idx))
                    cv2.imwrite(filename, (mask_16[v_idx] * 255).astype('uint8'))
                if self.layers >= 2:
                    filename = os.path.join(mask_dir, '%s_%03d_mask_32.png' % (var_name, start_idx + v_idx))
                    cv2.imwrite(filename, (mask_32[v_idx] * 255).astype('uint8'))
                if self.layers >= 1:
                    filename = os.path.join(mask_dir, '%s_%03d_mask_64.png' % (var_name, start_idx + v_idx))
                    cv2.imwrite(filename, (mask_64[v_idx] * 255).astype('uint8'))

        return output_dir

    def freeze_cont_stream(self):
        """
        freeze the content params during the motion learning
        """
        params = list(self.networks['generator'].contEnc.parameters()) + \
                 list(self.networks['generator'].contSampler.parameters()) + \
                 list(self.networks['generator'].videoDec.parameters()) + \
                 list(self.networks['generator'].contMotionStateGen.fc_cont1.parameters()) + \
                 list(self.networks['generator'].contMotionStateGen.fc_cont2.parameters())
        for param in params:
            param.requires_grad = False

    def free_cont_stream(self):
        """
        free the content params after the motion learning
        """
        params = list(self.networks['generator'].contEnc.parameters()) + \
                 list(self.networks['generator'].contSampler.parameters()) + \
                 list(self.networks['generator'].videoDec.parameters()) + \
                 list(self.networks['generator'].contMotionStateGen.fc_cont1.parameters()) + \
                 list(self.networks['generator'].contMotionStateGen.fc_cont2.parameters())
        for param in params:
            param.requires_grad = True

    def del_motion_dec(self):
        """
        delete the graph of the motion decoder
        """
        if self.task == 'pred':
            del self.losses['motion_dec']['l_dec_pred']
        elif self.task == 'video_generate':
            del self.losses['vid_dec']['l_dec']
        else:
            raise ValueError('task %s is not supported' % self.task)

    def del_cont_dec(self):
        """
        delete the graph of the content decoder
        """
        del self.losses['cont_dec']['l_dec_recons']

    def del_vid_dis(self):
        """
        delete the graph of the video discriminator
        """
        del self.losses['vid_dis']['l_dis']

    def del_img_dis(self):
        """
        delete the graph of the image discriminator
        """
        if self.task == 'pred':
            del self.losses['img_dis']['l_dis_pred']
        elif self.task == 'recons':
            del self.losses['img_dis']['l_dis_recons']
        else:
            raise ValueError('task %s is not supported' % self.task)

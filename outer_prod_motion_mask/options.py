import argparse
import os


class Options(object):
    def __init__(self):
        """
        Define the arguments for dataloader, paths and model arch
        """
        self.parser = argparse.ArgumentParser()

        # dataloader setting
        self.parser.add_argument('--dataset', type=str, help='the name of dataset')
        self.parser.add_argument('--dataroot', type=str, help='the path of the dataset')
        self.parser.add_argument('--textroot', type=str, help='the path of video list ')
        self.parser.add_argument('--every_nth', type=int, default=1, help='sample training videos using every nth frame')
        self.parser.add_argument('--video_length', type=int, default=10, help='length of the video')
        self.parser.add_argument('--image_size', type=int, default=64, help='resize all frames to this size')
        self.parser.add_argument('--n_channels', type=int, default=3, help='number of channels in the input data')
        self.parser.add_argument('--crop', action='store_true', help='crop the input from a random point')
        self.parser.add_argument('--miniclip', action='store_true', help='use the mini clip of dataset')

        # experiment setting
        self.parser.add_argument('--model', type=str, required=True, help='the model type [SGVAN|TwoStreamVAN]')
        self.parser.add_argument('--exp_name', type=str, required=True, help='the name of the experiment')
        self.parser.add_argument('--log_dir', type=str, default='/research/sunxm/video_generation/logs',
                                 help='the path of the log file')
        self.parser.add_argument('--checkpoint_dir', type=str, default='/research/sunxm/video_generation/checkpoints',
                                 help='the folder to save checkpoints')
        self.parser.add_argument('--output_dir', type=str, default='/research/sunxm/video_generation/results/',
                                 help='The folder to save the output')
        # model setting
        self.parser.add_argument('--batch_size', type=int, default=32, help='number of images in image batch')
        self.parser.add_argument('--motion_dim', type=int, default=64, help='the size of the motion vector')
        self.parser.add_argument('--cont_dim', type=int, default=512, help='the size of the content vector')
        self.parser.add_argument('--gf_dim', type=int, default=16, help='the number of model-base channels')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='the size of the latent space in vae')
        self.parser.add_argument('--which_iter', type=str, default='latest', help='load specific checkpoints')
        self.parser.add_argument('--use_bn', action='store_true',
                                 help='specify if using batch norm in the content stream')
        # # TwostreamVAN setting
        self.parser.add_argument('--no_mask', action='store_true', help='specified if not using motion mask')
        self.parser.add_argument('--joint', action='store_true', help='specified if the videoDec and kernelGen are trained jointly')
        self.parser.add_argument('--ac_kernel', type=int, default=5, help='the kernel size to apply on each scale')
        self.parser.add_argument('--kernel_layer', type=int, default=4, help='# layers to apply kernels')

    def parse(self):
        """
        Parse the arguments, find out the arguments for the generator arch
        """
        opt = self.parser.parse_args()
        opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.exp_name)
        opt.log_dir = os.path.join(opt.log_dir, opt.exp_name)
        opt.output_dir = os.path.join(opt.output_dir, opt.exp_name)

        # prepare the generator args
        gen_args = {}
        gen_args['n_channels'] = opt.n_channels
        gen_args['motion_dim'] = opt.motion_dim
        gen_args['cont_dim'] = opt.cont_dim
        gen_args['gf_dim'] = opt.gf_dim
        gen_args['latent_dim'] = opt.latent_dim
        gen_args['use_bn'] = opt.use_bn

        if opt.model == 'TwoStreamVAN':
            gen_args['ac_kernel'] = opt.ac_kernel
            gen_args['no_mask'] = opt.no_mask
            gen_args['joint'] = opt.joint
            gen_args['kernel_layer'] = opt.kernel_layer

        return opt, gen_args


class TrainOptions(Options):
    def __init__(self):
        """
        Define the arguments for the training process, and the loss weights
        """
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--resume', action='store_true', help='when specified the experiment is resumed')

        # training surveillance
        self.parser.add_argument('--print_freq', type=int, default=200, help='print losses every ? iterations')
        self.parser.add_argument('--save_freq', type=int, default=1000, help='the frequency to save models')
        self.parser.add_argument('--val_freq', type=int, default=1000, help='the frequency to do the validation')

        # training params
        self.parser.add_argument('--pretrain_iters', type=int, default=0,
                                 help='specify number of updates to pretrain content part')
        self.parser.add_argument('--total_iters', type=int, default=100000, help='specify number of updates to train')

        # loss weights
        self.parser.add_argument('--c_kl_start', type=int, required=True, help='kl loss weight for content stream')
        self.parser.add_argument('--c_kl_end', type=int, required=True, help='kl loss weight for content stream')
        self.parser.add_argument('--c_img_dis', type=int, default=10,
                                 help='the value of the image-level discriminator in contentVAE')
        self.parser.add_argument('--img_m_kl', type=int, default=7, help='kl loss weight for image-level motion VAE')
        self.parser.add_argument('--vid_m_kl_start', type=int, default=2,
                                 help='the value of video-level motion VAE at the begining')
        self.parser.add_argument('--vid_m_kl_end', type=int, default=20,
                                 help='the value of video_level motion VAE at the end')
        self.parser.add_argument('--xp_vs_xtilde', type=float, default=1,
                                 help='the ratio between xp and xtilde gan losses')
        self.parser.add_argument('--pred_scale_feat', type=float, default=100,
                                 help='the value of the scale feat loss in image-level training')
        self.parser.add_argument('--video_scale_feat', type=float, default=100,
                                 help='the value of the scale feat loss in video_level training')

        # content vs motion training ratio
        self.parser.add_argument('--cont_ratio_start', type=float, required=True,
                                 help='the ratio of the content at the beginning')
        self.parser.add_argument('--cont_ratio_end', type=float, required=True,
                                 help='the ratio of the content at the end')
        self.parser.add_argument('--cont_ratio_iter_start', type=int, required=True,
                                 help='the iter when the ratio of content training start to change')
        self.parser.add_argument('--cont_ratio_iter_end', type=int, required=True,
                                 help='the iter when the ratio of content training ends changing')

        # easier vs faster motion training ratio
        self.parser.add_argument('--motion_ratio_start', type=float, required=True,
                                 help='the ratio of the harder motion task at the beginning')
        self.parser.add_argument('--motion_ratio_end', type=float, required=True,
                                 help='the ratio of the harder motion task at the end')
        self.parser.add_argument('--motion_ratio_iter_start', type=int, required=True,
                                 help='the iter when the ratio of the harder motion task start to change')
        self.parser.add_argument('--motion_ratio_iter_end', type=int, required=True,
                                 help='the iter when the ratio of the harder motion task ends changing')

    def parse(self):
        """
        Parse the arguments, find out the arguments for the discriminator arch
        """
        opt, gen_args = super(TrainOptions, self).parse()

        # prepare the discriminator args
        dis_args = {}
        dis_args['n_channels'] = opt.n_channels

        # prepare the kl loss weights
        loss_weights = {}
        loss_weights['c_kl_start'] = opt.c_kl_start
        loss_weights['c_kl_end'] = opt.c_kl_end
        loss_weights['img_m_kl'] = opt.img_m_kl
        loss_weights['vid_m_kl_start'] = opt.vid_m_kl_start
        loss_weights['vid_m_kl_end'] = opt.vid_m_kl_end
        loss_weights['c_img_dis'] = opt.c_img_dis
        loss_weights['xp_vs_xtilde'] = opt.xp_vs_xtilde
        loss_weights['pred_scale_feat'] = opt.pred_scale_feat
        loss_weights['video_scale_feat'] = opt.video_scale_feat

        return opt, gen_args, dis_args, loss_weights


class TestOptions(Options):
    def __init__(self):
        """
        Define the arguments for the test process
        """
        super(TestOptions, self).__init__()
        self.parser.add_argument('--val_num', type=int, default=100, help='test times')
        self.parser.add_argument('--get_seq', action='store_true', help='save the video sequence at test')
        self.parser.add_argument('--get_mask', action='store_true', help='save the multi-layer masks at test')



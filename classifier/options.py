import argparse
import os


class Options(object):
    """Define and parse the common arguments for training and test"""
    def __init__(self):
        """Define the arguments"""
        self.parser = argparse.ArgumentParser()
        # exp settings
        self.parser.add_argument('--exp_name', type=str, help='the name of the experiment')
        self.parser.add_argument('--batch_size', type=int, default=32, help='number of images in image batch')

        # model setting
        self.parser.add_argument('--ndf', type=int, default=64, help='the feature dimension of the first layer')
        self.parser.add_argument('--model_is_actor', action='store_true', help='distinguish actor when specified')
        self.parser.add_argument('--data_is_actor', action='store_true', help='distinguish actor when specified')

        # dataset settings
        self.parser.add_argument('--dataset', type=str, help='the name of dataset')
        self.parser.add_argument('--dataroot', type=str, help='the path of the dataset')
        self.parser.add_argument('--textroot', type=str, help='the path of video list ')
        self.parser.add_argument('--every_nth', type=int, default=2,
                                 help='sample training videos using every nth frame')
        self.parser.add_argument('--video_length', type=int, default=10, help='length of the video')
        self.parser.add_argument('--image_size', type=int, default=64, help='resize all frames to this size')
        self.parser.add_argument('--n_channels', type=int, default=3, help='number of channels in the input data')
        self.parser.add_argument('--crop', action='store_true', help='crop the input from a random point')

        # save dirs
        self.parser.add_argument('--log_dir', type=str, default='/research/sunxm/classifier/logs',
                                 help='the path of the log file')
        self.parser.add_argument('--checkpoint_dir', type=str, default='/research/sunxm/classifier/checkpoints',
                                 help='the folder to save checkpoints')


class TrainOptions(Options):
    """Define and parse the training arguments"""

    def __init__(self):
        """Define the training arguments"""
        super(TrainOptions, self).__init__()
        # experiment surveillance
        self.parser.add_argument('--print_freq', type=int, default=10, help='print losses every ? iterations')
        self.parser.add_argument('--save_freq', type=int, default=500, help='the frequency to save models')
        self.parser.add_argument('--val_freq', type=int, default=100, help='the freqency to do the validation')

        # training setting
        self.parser.add_argument('--total_iters', type=int, default=100000, help='specify number of updates to train')
        self.parser.add_argument('--resume', action='store_true', help='when specified the experiment is resumed')
        self.parser.add_argument('--which_iter', type=str, default='latest', help='load specific checkpoints')
        self.parser.add_argument('--lr', type=float, help='the initial learning rate')
        self.parser.add_argument('--decay', type=float, help='the lr decay rate for one epoch')

    def parse(self):
        """
        Parsing the training arguments
        :return opt: the training arguments
        """
        opt = self.parser.parse_args()
        opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.exp_name)
        opt.log_dir = os.path.join(opt.log_dir, opt.exp_name)
        return opt


class TestOptions(Options):
    """Define and parse the test arguments"""
    def __init__(self):
        """Define the test arguments"""

        super(TestOptions, self).__init__()
        self.parser.add_argument('--ckpt_path', type=str, help='the pre-trained classifier path')
        self.parser.add_argument('--test_dataset', type=str, help='the name of dataset during test')
        self.parser.add_argument('--test_dataroot', type=str, help='the path of dataset during test')

    def parse(self):
        """
        Parse the test arguments
        :return opt: the test arguments
        """
        opt = self.parser.parse_args()
        return opt
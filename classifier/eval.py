import os
import sys
sys.path.insert(0, '..')
import functools

from torchvision import transforms
from torch.utils.data import DataLoader

from classifier.classifier_environ import Classifier_Environ
from utils.util import makedir
from data.weizmann_dataset import WeizmannDataset
from data.mug_dataset import MUGDataset, MUGDataset_2
from data.synaction_dataset import SynActionDataset
from data.video_dataset import VideoDataset
from data.generated_dataset import GeneratedDataset
from data.util import video_transform
from utils.util import listopt
from classifier.options import TestOptions
from classifier.metrics import *


def make_opt(args, output_dir, data_is_actor):
    """
    make the arguments for the classification from arguments of the main experiment
    :param args: class object, arguments of the main experiment
    :param output_dir: str, the path where the test videos are saved
    :param data_is_actor: bool, whether the test videos are generated with the actor knowledge
    :return opt: the arguments for the classification
    """
    class Opt(object):
        def __init__(self):
            self.n_channels = 3
            self.batch_size = 32
            self.ndf = 64
            self.video_length = args.video_length
            self.image_size = args.image_size
            self.dataset = 'Generated_video'
            self.dataroot = output_dir
            self.every_nth = args.every_nth
            self.log_dir = '/research/sunxm/classifier/logs'
            self.checkpoint_dir = '/research/sunxm/classifier/checkpoints'
            self.test_dataroot = args.dataroot

            if args.dataset == 'Weizmann':
                self.exp_name = 'weizmann_classifier_actor'
                self.ckpt_path = '/research/sunxm/classifier/weizmann/checkpoints/actorAction_best_model.pth.tar'
                self.model_is_actor = True
                self.test_dataset = 'Weizmann'
                self.textroot = '../videolist/Weizmann'
            elif args.dataset == 'MUG':
                self.exp_name = 'mug_classifier'
                self.ckpt_path = '/research/sunxm/classifier/mug/best_model.pth.tar'
                self.model_is_actor = False
                self.test_dataset = 'MUG'
                self.textroot = '../videolist/MUG'
            elif args.dataset == 'SynAction':
                self.exp_name = 'synaction_classifier'
                self.ckpt_path = '/research/sunxm/classifier/synaction/best_model.pth.tar'
                self.model_is_actor = False
                self.test_dataset = 'SynAction'
                self.textroot = '../videolist/SynAction'

            self.data_is_actor = data_is_actor
    opt = Opt()
    return opt


def eval(eval_args=None, output_dir=None, data_is_actor=False):

    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    if eval_args is None:
        opt = TestOptions().parse()
        # make up all dirs and print options
        listopt(opt)
        makedir(opt.log_dir)
        with open(os.path.join(opt.log_dir, 'test_options.txt'), 'w+') as f:
            listopt(opt, f)
    else:
        opt = make_opt(eval_args, output_dir, data_is_actor)

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    print('define image and video transformation')
    # get the dataloader
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x[:opt.n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    if opt.dataset == 'Generated_video':
        if opt.test_dataset == 'Weizmann':
            test_dataset = WeizmannDataset(opt.test_dataroot, opt.textroot, opt.video_length, opt.image_size,
                                     opt.every_nth, False, 'Test')
        elif opt.test_dataset == 'MUG':
            test_dataset = MUGDataset(opt.test_dataroot, opt.textroot, opt.video_length, opt.image_size,
                                           opt.every_nth, False, 'Test')
        elif opt.test_dataset == 'SynAction':
            test_dataset = SynActionDataset(opt.test_dataroot, opt.textroot, opt.video_length, opt.image_size, False, 'Test')
        else:
            raise NotImplementedError('%s is not implemented' % opt.dataset)
        action_set = test_dataset.action_set
        actor_set = test_dataset.actor_set
        valset = GeneratedDataset(opt.dataroot, opt.data_is_actor, actor_set, action_set)
        video_valset = valset
    else:
        if opt.dataset == 'Weizmann':
            print('create the video dataloader')
            # dataset, val_dataset = None, None
            valset = WeizmannDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size,
                                     opt.every_nth, False, 'Test')
        elif opt.dataset == 'MUG':
            valset = MUGDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False, 'Test')
        elif opt.dataset == 'MUG2':
            trainset = MUGDataset_2(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth,
                                    opt.crop, 'Train')
            print(trainset.action_set)
            valset = MUGDataset_2(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False, 'Test')
            print(valset.action_set)
        elif opt.dataset == 'SynAction':
            valset = SynActionDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, False, 'Test')
        else:
            raise NotImplementedError('%s is not implemented' % opt.dataset)
        video_valset = VideoDataset(valset, opt.video_length, every_nth=opt.every_nth, transform=video_transforms)
        action_set = valset.action_set
        actor_set = valset.actor_set
    video_val_loader = DataLoader(video_valset, batch_size=opt.batch_size, drop_last=False, num_workers=2, shuffle=False)

    # ********************************************************************
    # ******************** Create the Environment ************************
    # ********************************************************************
    # calculate the number of classes
    if opt.model_is_actor:
        num_class =len(action_set) * len(actor_set)
    else:
        num_class = len(action_set)

    print('create the environment')
    # build the training environment
    environ = Classifier_Environ(opt.n_channels, num_class, opt.log_dir, opt.checkpoint_dir, 1, opt.ndf)
    iter, _ = environ.load('latest', path=opt.ckpt_path)
    print('using iter %d' % iter)

    environ.eval()
    loss, gt_cat, pred_dist = [], [], []
    for idx, batch in enumerate(video_val_loader):
        if opt.data_is_actor and opt.model_is_actor:
            batch['categories'] = batch['categories'] + batch['actors'] * len(action_set)
        environ.set_inputs(batch)
        tmp_loss, tmp_gt, tmp_pred = environ.val()
        loss.append(tmp_loss)
        gt_cat.append(tmp_gt)
        pred_dist.append(tmp_pred)

    # get accuracy,intra_E, inter_E, class_intra_E
    pred_dist = np.concatenate(pred_dist, axis=0)
    pred_cat = np.argmax(pred_dist, axis=-1)
    gt_cat = np.concatenate(gt_cat)
    if opt.model_is_actor:
        pred_cat = pred_cat % len(action_set)
    if opt.data_is_actor:
        gt_cat = gt_cat % len(action_set)
    I_score, intra_E, inter_E, class_intra_E = quant(pred_dist, action_set)
    acc = float(len(gt_cat) - np.count_nonzero(pred_cat != gt_cat)) / len(gt_cat) * 100

    print('acc: %.3f%%, I_score: %.3f, intra_E: %.3f, inter_E: %.3f' % (acc, I_score, intra_E, inter_E))
    print('class intra-E', class_intra_E)
    print(action_set)

    metrics = {'acc': acc, 'I_score': I_score, 'intra_E': intra_E, 'inter_E': inter_E}
    return metrics


if __name__ == '__main__':
    eval()

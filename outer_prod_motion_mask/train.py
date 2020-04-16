import os
import sys
sys.path.insert(0, '..')
import functools
import time
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from outer_prod_motion_mask.twostreamvan_environ import TwoStreamVAN
from outer_prod_motion_mask.sgvan_environ import SGVAN
from outer_prod_motion_mask.options import TrainOptions
from classifier.eval import eval
from data.weizmann_dataset import WeizmannDataset
from data.mug_dataset import MUGDataset
from data.synaction_dataset import SynActionDataset
from data.video_dataset import VideoDataset
from data.util import video_transform
from utils.util import makedir, listopt
from utils.scheduler import Scheduler
import pdb


def main():
    torch.set_num_threads(1)
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    opt, gen_args, dis_args, loss_weights = TrainOptions().parse()
    makedir(opt.checkpoint_dir)
    makedir(opt.log_dir)
    makedir(opt.output_dir)
    listopt(opt)
    with open(os.path.join(opt.log_dir, 'train_opt.txt'), 'w+') as f:
        listopt(opt, f)

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x[:opt.n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    if opt.dataset == 'Weizmann':
        trainset = WeizmannDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, opt.crop,
                                   'Train', mini_clip=opt.miniclip)
        valset = WeizmannDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False,
                                       'Test', mini_clip=opt.miniclip)
    elif opt.dataset == 'MUG':
        trainset = MUGDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, opt.crop,
                              'Train')
        valset = MUGDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False, 'Test')
    elif opt.dataset == 'SynAction':
        trainset = SynActionDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.crop, 'Train')
        valset = SynActionDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, False, 'Test')
    else:
        raise NotImplementedError('%s dataset is not supported' % opt.dataset)

    # get the validate dataloader
    video_trainset = VideoDataset(trainset, opt.video_length, every_nth=opt.every_nth, transform=video_transforms)
    video_train_loader = DataLoader(video_trainset, batch_size=opt.batch_size, drop_last=True, num_workers=2, shuffle=True)

    video_valset = VideoDataset(valset, opt.video_length, every_nth=opt.every_nth, transform=video_transforms)
    video_val_loader = DataLoader(video_valset, batch_size=opt.batch_size, drop_last=True, num_workers=2, shuffle=False)

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************
    gen_args['num_categories'] = len(trainset.action_set)
    dis_args['num_categories'] = len(trainset.action_set)

    if opt.model == 'SGVAN':
        environ = SGVAN(gen_args, opt.checkpoint_dir, opt.log_dir, opt.output_dir, opt.video_length,
                        trainset.action_set, trainset.actor_set, is_eval=False, dis_args=dis_args,
                        loss_weights=loss_weights, pretrain_iters=opt.pretrain_iters)
    elif opt.model == 'TwoStreamVAN':
        environ = TwoStreamVAN(gen_args, opt.checkpoint_dir, opt.log_dir, opt.output_dir, opt.video_length,
                               trainset.action_set, trainset.actor_set, is_eval=False, dis_args=dis_args,
                               loss_weights=loss_weights, pretrain_iters=opt.pretrain_iters)
    else:
        raise ValueError('Model %s is not implemented' % opt.mode)

    current_iter = 0
    if opt.resume:
        current_iter = environ.load(opt.which_iter)
    else:
        environ.weight_init()
    environ.train()

    # ********************************************************************
    # ******************** Set the training ratio ************************
    # ********************************************************************
    # content vs motion
    cont_scheduler = Scheduler(opt.cont_ratio_start, opt.cont_ratio_end,
                               opt.cont_ratio_iter_start + opt.pretrain_iters, opt.cont_ratio_iter_end + opt.pretrain_iters,
                               mode='linear')
    # easier vs harder motion
    m_img_scheduler = Scheduler(opt.motion_ratio_start, opt.motion_ratio_end,
                                opt.motion_ratio_iter_start + opt.pretrain_iters, opt.motion_ratio_iter_end + opt.pretrain_iters,
                                mode='linear')

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    recons_c, pred_c, vid_c = 0, 0, 0
    video_enumerator = enumerate(video_train_loader)
    while current_iter < opt.total_iters:
        start_time = time.time()
        current_iter += 1
        batch_idx, batch = next(video_enumerator)
        environ.set_inputs(batch)

        if current_iter < opt.pretrain_iters:
            # ********************** Pre-train the Content Stream **************
            environ.optimize_recons_pretrain_parameters(current_iter)

            # print loss to the screen and save intermediate results to tensorboard
            if current_iter % opt.print_freq == 0:
                environ.print_loss(current_iter, start_time)
                environ.visual_batch(current_iter, name='%s_current_batch' % environ.task)

            # validation
            if current_iter % opt.val_freq == 0:
                environ.eval()
                # validation of the content generation
                for idx, batch in enumerate(video_val_loader):
                    environ.set_inputs(batch)
                    environ.reconstruct_forward(ae_mode='mean', is_eval=True)
                    if idx == 0:
                        environ.visual_batch(current_iter, name='val_recons')
                # save the current checkpoint
                environ.save('latest', current_iter)
                environ.train()
        else:
            # ********************* Jointly train the Content & Motion *************
            ep1 = cont_scheduler.get_value(current_iter)
            ep2 = m_img_scheduler.get_value(current_iter)
            recons = (random.random() > ep1)
            img_level = (random.random() > ep2)
            if recons:
                # content training
                recons_c += 1
                environ.optimize_recons_parameters(current_iter)
            else:
                if img_level:
                    # easier motion training
                    pred_c += 1
                    environ.optimize_pred_parameters()
                else:
                    # harder motion training
                    vid_c += 1
                    environ.optimize_vid_parameters(current_iter)

            # print loss to the screen and save intermediate results to tensorboard
            if current_iter % opt.print_freq == 0:
                environ.print_loss(current_iter, start_time)
                environ.visual_batch(current_iter, name='%s_current_batch' % environ.task)
                print('recons: %d, pred: %d, vid: %d' % (recons_c, pred_c, vid_c))
                recons_c, pred_c, vid_c = 0, 0, 0

            # validation and save checkpoint
            if current_iter % opt.val_freq == 0:
                environ.eval()
                for idx, batch in enumerate(video_val_loader):
                    environ.set_inputs(batch)

                    # content stream validation
                    environ.reconstruct_forward(ae_mode='mean', is_eval=True)
                    if idx == 0:
                        environ.visual_batch(current_iter, name='val_recons')

                    # easier motion stream validation
                    environ.predict_forward(ae_mode='mean', is_eval=True)
                    if idx == 0:
                        environ.visual_batch(current_iter, name='val_pred')

                    # harder motion stream validation
                    environ.video_forward(eplison=0, ae_mode='mean', is_eval=True)
                    if idx == 0:
                        environ.visual_batch(current_iter, name='val_video')

                # generate videos for different class
                for idx, cls_name in enumerate(valset.action_set):
                    environ.get_category(cls_id=idx)
                    output_dir = environ.full_test(idx, 32, 10, current_iter, cls_name)
                metrics = eval(opt, output_dir)
                environ.print_loss(current_iter, start_time, metrics)

                # remove the generated video
                rm_cmd = 'rm -r %s' % output_dir
                os.system(rm_cmd)

                # save the latest checkpoint
                environ.save('latest', current_iter)
                environ.train()

        # save the checkpoint
        if current_iter % opt.save_freq == 0:
            environ.save(current_iter, current_iter)

        # get a new enumerator
        if batch_idx == len(video_train_loader) - 1:
            video_enumerator = enumerate(video_train_loader)


if __name__ == '__main__':
    main()
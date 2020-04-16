import os
import sys
sys.path.insert(0, '..')
import functools

from torchvision import transforms
import time
from torch.utils.data import DataLoader

from classifier.classifier_environ import Classifier_Environ
from utils.util import makedir
from data.weizmann_dataset import WeizmannDataset
from data.mug_dataset import MUGDataset, MUGDataset_2
from data.synaction_dataset import SynActionDataset
from data.video_dataset import VideoDataset
from data.util import video_transform
from utils.util import listopt
from classifier.metrics import *
from classifier.options import TrainOptions


def main():
    # ************************************************************
    # ************** create folders and print options ************
    # ************************************************************
    opt = TrainOptions().parse()
    listopt(opt)
    print('create the directories')
    makedir(opt.checkpoint_dir)
    makedir(opt.log_dir)
    with open(os.path.join(opt.log_dir, 'train_options.txt'), 'w+') as f:
        listopt(opt, f)

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

    print('create the video dataloader')
    if opt.dataset == 'Weizmann':
        trainset = WeizmannDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, opt.crop, 'Train')
        valset = WeizmannDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False, 'Test')
    elif opt.dataset == 'MUG':
        trainset = MUGDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth,
                                   opt.crop, 'Train')
        valset = MUGDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False,
                                 'Test')
    elif opt.dataset == 'MUG2':
        trainset = MUGDataset_2(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth,
                                   opt.crop, 'Train')
        print(trainset.action_set)
        valset = MUGDataset_2(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False,
                                 'Test')
        print(valset.action_set)
    elif opt.dataset == 'SynAction':
        trainset = SynActionDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.crop, 'Train')
        valset = SynActionDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, False, 'Test')
    else:
        raise NotImplementedError('%s is not implemented' % opt.dataset)
    # get the validate dataloader
    video_trainset = VideoDataset(trainset, opt.video_length, every_nth=opt.every_nth, transform=video_transforms)
    video_train_loader = DataLoader(video_trainset, batch_size=opt.batch_size, drop_last=True, num_workers=2, shuffle=True)

    video_valset = VideoDataset(valset, opt.video_length, every_nth=opt.every_nth, transform=video_transforms)
    video_val_loader = DataLoader(video_valset, batch_size=opt.batch_size, drop_last=False, num_workers=2, shuffle=False)

    # ********************************************************************
    # ******************** Create the Environment ************************
    # ********************************************************************
    # calculate the number of classes
    if opt.model_is_actor:
        num_class = len(trainset.action_set) * len(trainset.actor_set)
    else:
        num_class = len(trainset.action_set)

    print('create the environment')
    environ = Classifier_Environ(opt.n_channels, num_class, opt.log_dir, opt.checkpoint_dir, opt.lr, opt.ndf)
    current_iter = 0

    # load the checkpoints
    if opt.resume:
        current_iter, opt.lr = environ.load(opt.which_iter)
    else:
        environ.weight_init()

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    action_set = trainset.action_set
    max_acc, max_I_score, best_iter = 0, 0, 0
    print('begin training')
    video_enumerator = enumerate(video_train_loader)
    while current_iter < opt.total_iters:
        start_time = time.time()
        environ.train()
        current_iter += 1

        batch_idx, batch = next(video_enumerator)
        # modify the gt category if the model needs to distinguish actors
        if opt.data_is_actor:
            batch['categories'] = batch['categories'] + batch['actors'] * len(action_set)
        environ.set_inputs(batch)

        environ.optimize()

        # print losses
        if current_iter % opt.print_freq == 0:
            environ.print_loss(current_iter, start_time)

        # validation
        if current_iter % opt.val_freq == 0:
            environ.eval()
            loss, gt_cat, pred_dist = [], [], []

            # go through the validation set
            for idx, batch in enumerate(video_val_loader):
                environ.set_inputs(batch)
                if opt.data_is_actor:
                    batch['categories'] = batch['categories'] + batch['actors'] * len(action_set)
                tmp_loss, tmp_gt, tmp_pred = environ.val()
                loss.append(tmp_loss)
                gt_cat.append(tmp_gt)
                pred_dist.append(tmp_pred)

            pred_dist = np.concatenate(pred_dist, axis=0)
            pred_cat = np.argmax(pred_dist, axis=-1)
            gt_cat = np.concatenate(gt_cat)

            # calculate the metrics
            I_score, intra_E, inter_E, class_intra_E = quant(pred_dist, trainset.action_set)
            acc = float(len(gt_cat) - np.count_nonzero(pred_cat != gt_cat))/len(gt_cat) * 100
            loss = {'val_loss': np.mean(loss), 'acc': acc, 'I_score': I_score, 'intra_E': intra_E, 'inter_E': inter_E}
            environ.print_loss(current_iter, start_time, loss=loss)

            # save the checkpoint if the current gives the best performance
            if acc >= max_acc and I_score >= max_I_score:
                max_acc = acc
                max_I_score = I_score
                environ.save('best', current_iter)
                best_iter = current_iter
            print('max_I_score: %.3f, max_acc: %.3f, best iter: %d' % (max_I_score, max_acc, best_iter))
            environ.save(current_iter, current_iter)

        # save the current iteration
        if current_iter % opt.save_freq == 0:
            environ.save('latest', current_iter)

        # adjust the learning rate
        if batch_idx == len(video_train_loader) - 1:
            video_enumerator = enumerate(video_train_loader)
            opt.lr = opt.lr * opt.decay
            environ.adjust_learning_rate(opt.lr)


if __name__ == '__main__':
    main()

import os
import sys
sys.path.insert(0, '..')
import functools

from torchvision import transforms
from torch.utils.data import DataLoader

from outer_prod_motion_mask.options import TestOptions
from outer_prod_motion_mask.sgvan_environ import SGVAN
from outer_prod_motion_mask.twostreamvan_environ import TwoStreamVAN

from utils.util import makedir, listopt
from data.weizmann_dataset import WeizmannDataset
from data.mug_dataset import MUGDataset
from data.video_dataset import VideoDataset
from data.util import video_transform
from data.synaction_dataset import SynActionDataset

from classifier.eval import eval
import tqdm


def print_dict(dict, dict_name):
    line = dict_name
    for k, v in dict.items():
        line += '%s:%03f' % (k, v)
    return line


def main():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    opt, gen_args = TestOptions().parse()
    makedir(opt.output_dir)
    makedir(opt.log_dir)
    listopt(opt)
    with open(os.path.join(opt.log_dir, 'test_opt.txt'), 'w+') as f:
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
        valset = WeizmannDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False,
                                'Test', mini_clip=opt.miniclip)
    elif opt.dataset == 'MUG':
        valset = MUGDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, opt.every_nth, False, 'Test')
    elif opt.dataset == 'SynAction':
        valset = SynActionDataset(opt.dataroot, opt.textroot, opt.video_length, opt.image_size, False, 'Test')
    else:
        raise NotImplementedError('%s dataset is not supported' % opt.dataset)

    # get the validate dataloader
    video_valset = VideoDataset(valset, opt.video_length, every_nth=opt.every_nth, transform=video_transforms)
    video_val_loader = DataLoader(video_valset, batch_size=opt.batch_size, drop_last=False, num_workers=2, shuffle=False)

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************
    gen_args['num_categories'] = len(valset.action_set)
    if opt.model == 'SGVAN':
        environ = SGVAN(gen_args, opt.checkpoint_dir, opt.log_dir, opt.output_dir, opt.video_length,
                               valset.action_set, valset.actor_set, is_eval=True)
    elif opt.model == 'TwoStreamVAN':
        environ = TwoStreamVAN(gen_args, opt.checkpoint_dir, opt.log_dir, opt.output_dir, opt.video_length,
                               valset.action_set, valset.actor_set, is_eval=True)
    else:
        raise ValueError('Model %s is not implemented' % opt.mode)

    current_iter = environ.load(opt.which_iter, is_eval=True)
    environ.eval()

    # ********************************************************************
    # ***************************  Full test  ****************************
    # ********************************************************************
    rm_npy = True
    result_file = os.path.join(opt.log_dir, 'results.txt')
    for idx, cls_name in enumerate(valset.action_set):
        for c in range(10):
            prefix = 'none_%s' % cls_name
            if opt.model == 'SGVAN':
                output_dir = environ.full_test(idx, 90, opt.video_length, current_iter, var_name=prefix, start_idx=c * 90,
                                               is_eval=True, rm_npy=rm_npy)
            elif opt.model == 'TwoStreamVAN':
                output_dir = environ.full_test(idx, 90, opt.video_length, current_iter, start_idx=c*90, var_name=prefix,
                                               is_eval=True, rm_npy=rm_npy, get_seq=opt.get_seq, get_mask=opt.get_mask)
            else:
                raise ValueError('Model %s is not implemented' % opt.model)
            rm_npy = False
    full_metrics = eval(opt, output_dir)
    with open(result_file, 'a') as f:
        f.writelines(print_dict(full_metrics, 'full_metric') + '\n')

    # ********************************************************************
    # ************************  Conditional Test *************************
    # ********************************************************************
    # # provide the first frame
    for c in tqdm.tqdm(range(opt.val_num)):
        for idx, batch in enumerate(video_val_loader):
            environ.set_inputs(batch)
            environ.video_forward(eplison=0, ae_mode='mean', is_eval=True)
            names = batch['names']
            output_dir = environ.save_batch(current_iter, names=names, start_idx=c)
    metrics = eval(opt, output_dir, data_is_actor=True)

    print(full_metrics)
    print(metrics)
    with open(result_file, 'a') as f:
        f.writelines(print_dict(metrics, 'avg_metric') + '\n')


if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vu
import os
from torch import nn

import pdb


def show_batch(batch):
    normed = batch * 0.5 + 0.5
    is_video_batch = len(normed.size()) > 4

    if is_video_batch:
        rows = [vu.make_grid(b.permute(1, 0, 2, 3), nrow=b.size(1)).numpy() for b in normed]
        im = np.concatenate(rows, axis=1)
    else:
        im = vu.make_grid(normed).numpy()

    im = im.transpose((1, 2, 0))

    plt.imshow(im)
    plt.show(block=True)


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def listopt(opt, f=None):
    """Pretty-print a given namespace either to console or to a file.

    :param opt: A namespace
    :param f: The file descriptor to write to. If None, write to console
    """
    args = vars(opt)

    if f is not None:
        f.write('------------ Options -------------\n')
    else:
        print('------------ Options -------------')

    for k, v in sorted(args.items()):
        if f is not None:
            f.write('%s: %s\n' % (str(k), str(v)))
        else:
            print('%s: %s' % (str(k), str(v)))

    if f is not None:
        f.write('-------------- End ----------------\n')
    else:
        print('-------------- End ----------------')


def print_current_errors(log_name, update, errors, t):
    message = 'update: %d, time: %.3f ' % (update, t)
    for k, v in errors.items():
        if k.startswith('Update'):
            message += '%s: %s ' % (k, str(v))
        else:
            message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message)


def images_to_visual(tensor):
    generated = torch.clamp(tensor.data.cpu(), min=-1, max=1)
    generated = (generated + 1) / 2
    return generated


def videos_to_visual(tensor):
    # [batch, c, t, h, w] -> [batch, t, c, h, w] -> [batch * t, c, h, w]
    s = tensor.data.size()
    generated = tensor.data.permute(0, 2, 1, 3, 4).view(-1, s[1], s[3], s[4])
    generated = (generated + 1) / 2
    return generated


def videos_to_numpy(tensor):
    # [batch, c, t, h, w] -> [batch, t, h, w, c]
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 4, 1).clip(-1, 1)
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def rgb2gray(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = None
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            gray_ = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
            gray = torch.unsqueeze(gray_, 0)
        elif image.dim() == 4:
            gray_ = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
            gray = torch.unsqueeze(gray_, 1)
        else:
            raise ValueError('The dimension of tensor is %d not supported in rgb2gray' % image.dim())
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 3:
                gray_ = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
                gray = np.expand_dims(gray_, 0)
            else:
                gray_ = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
                gray = np.expand_dims(gray_, -1)
        elif image.ndim == 4:
            gray_ = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
            gray = np.expand_dims(gray_, 1)
        else:
            raise ValueError('The dimension of np.ndarray is %d not supported in rgb2gray' % image.ndim)
    return gray


def one_hot(category_labels, num_categories):
    '''

    :param category_labels: a np.ndarray or a tensor with size [batch_size, ]
    :return: a tensor with size [batch_size, num_categories]
    '''
    if isinstance(category_labels, torch.Tensor):
        labels = category_labels.cpu().numpy()
    else:
        labels = category_labels
    num_samples = labels.shape[0]
    one_hot_labels = np.zeros((num_samples, num_categories), dtype=np.float32)  # [num_samples. dim_z_category]
    one_hot_labels[np.arange(num_samples), labels] = 1
    one_hot_labels = torch.from_numpy(one_hot_labels)

    if torch.cuda.is_available():
        one_hot_labels = one_hot_labels.cuda()
    return one_hot_labels


def compute_grad(inputs):
    """
    :param inputs: a tensor with size [batch_size, c, h, w]
    :return: a tensor with size [batch_size, 2c, h, w]
    """
    batch_size, n_channels, h, w = int(inputs.size()[0]), int(inputs.size()[1]), int(inputs.size()[2]), int(inputs.size()[3])
    grad = torch.zeros((batch_size, 2 * n_channels, h, w))
    grad[:, : n_channels, :-1] = (inputs[:, :, :-1] - inputs[:, :, 1:])/2
    grad[:, n_channels:, :, :-1] = (inputs[:, :, :, :-1] - inputs[:, :, :, 1:])/2
    if torch.cuda.is_available():
        grad = grad.cuda()
    return grad


class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Conv3d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)

import sys
sys.path.insert(0, '..')
from classifier.network import *
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn.init as init
import os
from utils.util import Initializer
from tensorboardX import SummaryWriter
import time
import pdb


class Classifier_Environ(object):
    """
    The training and test environment for the classification
    """
    def __init__(self, n_channels, num_class, log_dir, checkpoint_dir, lr, ndf, is_train=True):
        """
        :param n_channels: int, the color channels for the input channel
        :param num_class: int, the number of classes in the dataset
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param ndf: int, the basic channels of the model
        :param is_train: bool, specify during the training
        """
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.classifier = Classifier(n_channels, num_class, ndf=ndf)
        if torch.cuda.is_available():
            self.classifier.cuda()

        # define softmax function
        self.softmax = torch.nn.Softmax(dim=-1)

        # define loss
        self.category_criterion = nn.CrossEntropyLoss()

        if is_train:
            # define optimizer
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.00001)

            # define summary writer
            self.writer = SummaryWriter(log_dir=log_dir)

    def weight_init(self):
        """
        Initialize the network
        """
        Initializer.initialize(model=self.classifier, initialization=init.xavier_uniform_, gain=init.calculate_gain('relu'))

    def set_inputs(self, batch):
        """
        :param batch: {'images': a tensor [batch_size, c, video_len, h, w], 'categories': np.ndarray [batch_size,]}
        """
        if torch.cuda.is_available():
            self.images = Variable(batch['images']).cuda()
            self.categories = Variable(batch['categories']).cuda()
        else:
            self.images = Variable(batch['images'])
            self.categories = Variable(batch['categories'])

    def forward(self):
        """
        Forward method
        """
        self.output = self.classifier(self.images)

    def get_loss(self):
        """
        Calculate the classification loss
        """
        self.loss = self.category_criterion(self.output, self.categories)

    def backward(self):
        """
        Backward method
        """
        self.optimizer.zero_grad()
        self.get_loss()
        self.loss.backward()
        self.optimizer.step()

    def val(self):
        """
        Get the classification loss, the predicted category and the prediction distribution
        """
        self.forward()
        self.get_loss()
        distribution = self.softmax(self.output)
        return self.loss.data, self.categories.data.cpu().numpy(), distribution.data.cpu().numpy()

    def optimize(self):
        """
        The overall optimization in the training
        """
        self.forward()
        self.backward()

    def print_loss(self, current_iter, start_time, loss=None):
        """
        Print the loss on the screen and the tensorboard
        :param current_iter: int, the current iteration
        :param start_time: time, the start time of this iteration
        :param loss: dict, specify when printing the metrics
        :return:
        """
        if loss is None:
            loss = {'training_loss': self.loss.data}
        message = 'update: %d, time: %.3f ' % (current_iter, time.time() - start_time)

        for k, v in loss.items():
            # save to tensorboard
            self.writer.add_scalar(k, v, current_iter)
            message += '%s: %.3f ' % (k, v)

        # print on the screen
        print(message)

        # save in the txt file
        log_name = os.path.join(self.log_dir, 'loss.txt')
        with open(log_name, 'a') as log_file:
            log_file.write('%s \n' % message)

    def get_current_state(self, current_iter):
        """
        Get the current params of classifier, optimizer and iteration
        :param current_iter: int, the current iteration
        :return current dict: dict, containing the current params of classifier, optimizer and iteration
        """
        current_state = {}
        current_state['classifer'] = self.classifier.state_dict()
        current_state['optimizer'] = self.optimizer.state_dict()
        current_state['iter'] = current_iter
        return current_state

    def save(self, label, current_iter):
        """
        Save the current checkpoint
        :param label: str, the label for the loading checkpoint
        :param current_iter: int, the current iteration
        """
        current_state = self.get_current_state(current_iter)
        save_filename = '%s_model.pth.tar' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(current_state, save_path)

    def load_snapshot(self, snapshot):
        """
        :param snapshot: dict, saving the params of classifier, optimizer and the iteration
        :return iter: the current iteration of the snapshot; lr: float, the current learning rate
        """
        self.classifier.load_state_dict(snapshot['classifer'])
        self.optimizer.load_state_dict(snapshot['optimizer'])
        lr = self.optimizer.param_groups[0]['lr']
        return snapshot['iter'], lr

    def load(self, label, path=None):
        """
        load the checkpoint
        :param label: str, the label for the loading checkpoint
        :param path: str, specify if knowing the checkpoint path
        """
        if path is None:
            save_filename = '%s_model.pth.tar' % label
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        else:
            save_path = path
        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            snapshot = torch.load(save_path)
            return self.load_snapshot(snapshot)
        else:
            raise ValueError('snapshot %s does not exist' % save_path)

    def adjust_learning_rate(self, new_lr):
        """Sets the learning rate to the new_lr"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self):
        """
        Change to the training mode
        """
        self.classifier.train()

    def eval(self):
        """
        Change to the eval mode
        """
        self.classifier.eval()

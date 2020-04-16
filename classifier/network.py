from torch import nn


class Classifier(nn.Module):
    """
    The classifier module
    """
    def __init__(self, n_channels, num_class, ndf=16):
        """
        Network Architecture
        :param n_channels: int, the color channels of the input video
        :param num_class: int, the number of the classes
        :param ndf: int, the basic channels of the model
        """
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, num_class, 4, 1, padding=(1, 0, 0), bias=False),
        )

    def forward(self, input):
        """
        Forward method
        :param input: Variable, the input video with the size [batch, c, t, h, w]
        :return h: Variable, with the size [batch, num_class]
        """
        h = self.main(input).squeeze()
        return h


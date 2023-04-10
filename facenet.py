import torch
from torch import nn, optim
from torch.nn import functional as F
from facenet_utils import Inception


class FaceNetNN2(nn.Module):
    '''
    NN2 version of the FaceNet model.
    Reference: https://arxiv.org/abs/1503.03832.
    '''

    def __init__(self,
            loss_fn = nn.TripletMarginLoss,
            loss_fn_margin: float = 0.2,
            optim_fn = optim.SGD,
            optim_fn_lr: float = 0.05
        ):
        super(FaceNetNN2, self).__init__()
        
        # layers
        self.conv1 = nn.Conv2d(3, 64, (7,7), 2, padding=3)
        self.maxpool1 = nn.MaxPool2d((3,3), 2, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.inception_2 = Inception(64, None, 64, 192, None, None, None)
        self.norm2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d((3,3), 2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, ('m', 32, 1))
        self.inception3b = Inception(256, 64, 96, 128, 32, 64, ('L2', 64, 1))
        self.inception3c = Inception(320, None, 128, (256, 2), 32, (64, 2), ('m', None, 2))
        self.inception4a = Inception(640, 256, 96, 192, 32, 64, ('L2', 128, 1))
        self.inception4b = Inception(640, 224, 112, 224, 32, 64, ('L2', 128, 1))
        self.inception4c = Inception(640, 192, 128, 256, 32, 64, ('L2', 128, 1))
        self.inception4d = Inception(640, 160, 144, 288, 32, 64, ('L2', 128, 1))
        self.inception4e = Inception(640, None, 160, (256, 2), 64, (128, 2), ('m', None, 2))
        self.inception5a = Inception(1024, 384, 192, 384, 48, 128, ('L2', 128, 1))
        self.inception5b = Inception(1024, 384, 192, 384, 48, 128, ('m', 128, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fully_conn = nn.Linear(1024, 128) # bias=True
        # L2 normalization (see forward)

        # loss and optim functions
        self.loss_fn = loss_fn(margin=loss_fn_margin)
        self.optim_fn = optim_fn(self.parameters(), lr=optim_fn_lr)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.norm1(x)
        x = self.inception_2(x)
        x = self.norm2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fully_conn(x)
        x = F.normalize(x)
        return x

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.googlenet import BasicConv2d


# for Inception (from below)

class L2Pool(nn.Module):
    '''
    L2 Pooling module.
    Reference: https://discuss.pytorch.org/t/how-do-i-create-an-l2-pooling-2d-layer/105562/5.
    '''

    def __init__(self, kernel_size: tuple, *args, **kwargs):
        super(L2Pool, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, *args, **kwargs)
        self.n = kernel_size[0] * kernel_size[1]
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.avg_pool(x**2) * self.n)


# for FaceNetNN2

class Inception(nn.Module):
    '''
    Inception module template for FaceNetNN2 model from GoogLeNet (Inception) model.
    Reference: https://pytorch.org/hub/pytorch_vision_googlenet/
    '''

    def __init__(self,
            in_channels: int,
            filters1x1: int | None = None,
            filters3x3_reduce: int | None = None,
            filters3x3: int | tuple | None = None,
            filters5x5_reduce: int | None = None,
            filters5x5: int | tuple | None = None,
            pool_proj_p: tuple | None = None
        ):
        super(Inception, self).__init__()

        if type(filters3x3) == tuple:
            stride_3x3 = filters3x3[1]
            filters3x3 = filters3x3[0]
        else:
            stride_3x3 = 1

        if type(filters5x5) == tuple:
            stride_5x5 = filters5x5[1]
            filters5x5 = filters5x5[0]
        else:
            stride_5x5 = 1

        # layers
        
        if filters1x1 != None:
            self.branch1 = BasicConv2d(in_channels, filters1x1, kernel_size=(1,1))
        else:
            self.branch1 = None

        if filters3x3 != None:
            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, filters3x3_reduce, kernel_size=(1,1)),
                BasicConv2d(filters3x3_reduce, filters3x3, kernel_size=(3,3), stride=stride_3x3, padding=1)
            )
        else:
            self.branch2 = None
        
        if filters5x5 != None:
            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, filters5x5_reduce, kernel_size=(1,1)),
                # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
                # Please see https://github.com/pytorch/vision/issues/906 for details.
                BasicConv2d(filters5x5_reduce, filters5x5, kernel_size=(3,3), stride=stride_5x5, padding=1)
            )
        else:
            self.branch3 = None
        
        if pool_proj_p != None:
            
            if pool_proj_p[0] == 'm':
                if pool_proj_p[1] != None:
                    self.branch4 = nn.Sequential(
                        nn.MaxPool2d((3,3), stride=pool_proj_p[2], padding=1),
                        BasicConv2d(in_channels, pool_proj_p[1], kernel_size=(1,1))
                    )
                else:
                    self.branch4 = nn.MaxPool2d((3,3), stride=pool_proj_p[2], padding=1)

            elif pool_proj_p[0] == 'L2':
                if pool_proj_p[1] != None:
                    self.branch4 = nn.Sequential(
                        L2Pool((3,3), stride=pool_proj_p[2], padding=1),
                        BasicConv2d(in_channels, pool_proj_p[1], kernel_size=(1,1))
                    )
                else:
                    self.branch4 = L2Pool((3,3), stride=pool_proj_p[2], padding=1)
            
            else:
                raise Exception("The type of pooling inside Inception is not defined.")    
        else:
            self.branch4 = None
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_branch1 = self.branch1(x) if self.branch1 != None else None
        x_branch2 = self.branch2(x) if self.branch2 != None else None
        x_branch3 = self.branch3(x) if self.branch3 != None else None
        x_branch4 = self.branch4(x) if self.branch4 != None else None
        
        x = torch.cat([x_branch for x_branch in (x_branch1, x_branch2, x_branch3, x_branch4) if x_branch != None], dim=1)

        return x
    
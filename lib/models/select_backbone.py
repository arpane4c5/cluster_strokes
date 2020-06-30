from models.resnet_2d3d import * 
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
#from torchsummary import summary
import torch.nn.functional as F
from torch.autograd import Variable

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        image_modules = list(models.video.r3d_18(pretrained=False, progress=True).children())[:-2] #all layer expect last layer
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        a = self.modelA(image)
        a = a.view(2, 2, 512, 7, 7)
        a = F.avg_pool3d(a, (2, 4, 4), stride=(4, 1, 1))
        a = a.view(2, 128, 2, 4, 4)
        return a

def select_resnet(network, track_running_stats=True,):
    param = {'feature_size': 1024}
    if network == 'resnet18':
        model = resnet18_2d3d_full(track_running_stats=track_running_stats)
        param['feature_size'] = 256
    elif network == 'full3Dresnet18':
        model = MyModel()
        param['feature_size'] = 128
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=track_running_stats)
        param['feature_size'] = 256 
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet200':
        model = resnet200_2d3d_full(track_running_stats=track_running_stats)
    else: raise IOError('model type is wrong')

    return model, param

if __name__ == '__main__':
    x_image = Variable(torch.randn(1, 3, 16, 112, 112))
    
    model = MyModel()
    output = model(x_image)
    print(output.shape)
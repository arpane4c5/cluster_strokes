import os
import torch
from models.select_backbone import select_resnet
import torchvision
from torch import nn

class Img2Vec():

    def __init__(self, model_path='checkpoints/batsman-fine-tuning-dict-wholeDataset.pt', 
                 pretrained=True):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if pretrained:
            self.model = torchvision.models.resnet50(pretrained=True)
#            self.model = nn.Sequential(*list(pretrained_model.children())[:-1])
        else:
            if torch.cuda.is_available():
                self.model = torch.load(model_path) # because the model was trained on a cuda machine
            else:
                self.model = torch.load(model_path, map_location='cpu')

        self.extraction_layer = self.model._modules.get('avgpool')
        self.layer_output_size = 2048

        self.model = self.model.to(self.device)
        self.model.eval()


    def get_vec(self, image):

        image = image.to(self.device)

        num_imgs = image.size(0)

        my_embedding = torch.zeros(num_imgs, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        return my_embedding.view(num_imgs, -1)
    
    
class Clip2Vec():

    def __init__(self, model_path=None, nclasses=5):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #self.backbone, params = select_resnet(network, track_running_stats=False)
        
        self.model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        
        if model_path is not None:    
            inp_feat_size = self.model.fc.in_features
            self.model.fc = nn.Linear(inp_feat_size, nclasses)
            self.model = self.model.to(self.device)
    
            # load checkpoint:
            if os.path.isfile(model_path):
                print("Loading weights : {}".format(model_path))
                self.model.load_state_dict(torch.load(model_path))
        # Evaluate using model
        for ft in self.model.parameters():
            ft.requires_grad = False
        self.extraction_layer = self.model._modules.get('avgpool')
        
#        self.backbone = nn.Sequential(*image_modules)
        params = {'feature_size' : self.model._modules.get('fc').in_features}
        if model_path is not None:
            self.backbone = torch.load(model_path)
#        if torch.cuda.is_available():
#            self.backbone = torch.load(model_path) # because the model was trained on a cuda machine
#        else:
#            self.backbone = torch.load(model_path, map_location='cpu')

#        self.extraction_layer = self.backbone.layer4[1].bn2
        self.layer_output_size = params['feature_size']

        self.model = self.model.to(self.device)
        self.model.eval()


    def get_vec(self, clip):

        clip = clip.to(self.device)

        num_clips = clip.size(0)

        my_embedding = torch.zeros(num_clips, self.layer_output_size, 1, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(clip)
        h.remove()
#        return h_x.view(num_clips, self.layer_output_size, 1, 1)

        return my_embedding.view(num_clips, 1, -1)
    
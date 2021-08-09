
import sys
sys.path.insert(0, '../../../../pytorch-i3d')
import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np 
import math 
import torchvision
from torch.autograd import Variable
from pytorch_i3d import InceptionI3d


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    
    def get_vec(self, src):

        num_clips, seq_size = src.size(0), src.size(1)

        my_embedding = torch.zeros(num_clips, seq_size, self.ninp)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        extraction_layer = self.transformer_encoder
        h = extraction_layer.register_forward_hook(copy_data)
        h_x = self.forward(src)
        h.remove()

        return my_embedding


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModelSA(nn.Module):

    def __init__(self, inp_ft_dim, enc_dim, output_dim, nhead, nhid, nlayers, 
                 dropout=0.5):
        super(TransformerModelSA, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.inp_ft_dim = inp_ft_dim
        self.enc_dim = enc_dim      # for ninp
        self.output_dim = output_dim
        self.pos_encoder = PositionalEncoding(enc_dim, dropout)
        encoder_layers = TransformerEncoderLayer(enc_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(inp_ft_dim, enc_dim)
        self.enc_dim = enc_dim
        self.decoder = nn.Linear(enc_dim, output_dim)
#        self.fc = nn.Linear(ntoken, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
#        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
#        self.fc.bias.data.zero_()
#        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.enc_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
#        output = self.fc(output)
        return output
    
    def get_vec(self, src):

        num_clips, seq_size = src.size(0), src.size(1)

        my_embedding = torch.zeros(num_clips, seq_size, self.enc_dim)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        extraction_layer = self.transformer_encoder
        h = extraction_layer.register_forward_hook(copy_data)
        h_x = self.forward(src)
        h.remove()
        return my_embedding
    
# base is resnet
# Tail is the main transormer network 
class Semi_Transformer(nn.Module):
    def __init__(self, num_classes, seq_len, inp_ft_dim=200, enc_dim=200, 
                 nhead=2, nhid=2, nlayers=2):
        super(Semi_Transformer, self).__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
#        self.tail = Tail(num_classes, seq_len)
        self.transf = TransformerModelSA(inp_ft_dim, enc_dim, num_classes, nhead, 
                                         nhid, nlayers, dropout=0.2)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3))
        x = x.view(b, t, -1)
        x = x.permute(1, 0, 2)      # convert to (SEQ, BATCH, DIM)
        x = self.transf(x)
        x = x.permute(1, 0, 2)      # convert to (BATCH, SEQ, DIM)
        return x
        # x: (b,t,2048,7,4)
#        return self.tail(x, b , t )
        
        

if __name__ == '__main__':
    
#    pretrained_model_path = '/home/arpan/VisionWorkspace/pytorch-i3d/models/rgb_imagenet.pt'
#    temporal_len = 32
#    i3dfeat_seq_len = temporal_len // 8  # for Mixed_5c layer T/8 (2,7,7) for (16,224,224)
#    model = Semi_Transformer(num_classes=8 , seq_len = i3dfeat_seq_len, 
#                             model = pretrained_model_path)
#    
#    model = model.cuda()    
#    print("Loaded model ....")
#    inp = torch.randn((2, 3, temporal_len, 224, 224))
#    inp = inp.cuda()
#    out = model(inp)
#    print("Executed !")
    
    max_seq_len = 8
    ntokens = 8192
    enc_dim = 200
    model = Semi_Transformer(num_classes=8 , seq_len = max_seq_len, inp_ft_dim=ntokens, 
                             enc_dim=enc_dim)
    model = model.cuda()
    print("Loaded model ....")
    inp = torch.randn((4, 8, 3, 56, 56))
    inp = inp.cuda()
    out = model(inp)
    print("Executed !")
    # model.transf.transformer_encoder.layers[0].self_attn.out_proj.weight
    
    
#    ntokens = 200
#    emsize = 200 # embedding dimension
#    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
#    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#    nhead = 2 # the number of heads in the multiheadattention models    
    
    
###############################################################################

#class Semi_Transformer(nn.Module):
#    def __init__(self, num_classes, seq_len, model=None):
#        super(Semi_Transformer, self).__init__()
#        self.num_classes = num_classes
#        self.seq_len = seq_len
##        resnet50 = torchvision.models.resnet50(pretrained=True)
##        self.base = nn.Sequential(*list(resnet50.children())[:-2])
#        self.i3d = InceptionI3d(400, in_channels=3)
#        if model is not None:
#            print("Loading model : {}".format(model))
#            self.i3d.load_state_dict(torch.load(model))
##        self.base = nn.Sequential(*list(i3d.children())[:-2])
##        self.tail = Tail(num_classes, seq_len, head=16)
#        self.transf = TransformerModelSA()
#
#    def forward(self, x):
#        b = x.size(0)
##        t = x.size(1)
##        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
#        t = x.size(2)
#        x = self.i3d.extract_layer_features(x, "Mixed_5c")  # Mixed_5c
#        # output = T/4, H/16, W/16 (8, 14, 14) 
#        # Pass this to an RPN to generate person BBoxes
#        x = x.transpose(1, 2)
#        x = x.reshape(b*x.size(1), x.size(2), x.size(3), x.size(4))
#        # x: (8,832,14,14) (t,dim,h,w) for Mixed_4f,  (4, 1024, 7, 7) for Mixed_5c
##        return self.tail(x, b, x.size(0) // b)

## Standard 2 layerd FFN of transformer
#class FeedForward(nn.Module):
#    def __init__(self, d_model, d_ff=2048, dropout = 0.3):
#        super(FeedForward, self).__init__() 
#        # We set d_ff as a default to 2048
#        self.linear_1 = nn.Linear(d_model, d_ff)
#        self.dropout = nn.Dropout(dropout)
#        self.linear_2 = nn.Linear(d_ff, d_model)
#        
#        nn.init.normal_(self.linear_1.weight, std=0.001)  
#        nn.init.normal_(self.linear_2.weight, std=0.001)  
#
#    def forward(self, x):
#        x = self.dropout(F.relu(self.linear_1(x)))
#        x = self.linear_2(x)
#        return x
#
## standard NORM layer of Transformer
#class Norm(nn.Module):
#    def __init__(self, d_model, eps = 1e-6, trainable=True):
#        super(Norm, self).__init__()
#        self.size = d_model
#        # create two learnable parameters to calibrate normalisation
#        if trainable:
#            self.alpha = nn.Parameter(torch.ones(self.size))
#            self.bias = nn.Parameter(torch.zeros(self.size))
#        else:
#            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
#            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
#        self.eps = eps
#    def forward(self, x):
#        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
#        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
#        return norm
#
#
## Standard positional encoding (addition/ concat both are valid) 
#class PositionalEncoder(nn.Module):
#    def __init__(self, d_model, max_seq_len = 80):
#        super(PositionalEncoder, self).__init__()
#        self.d_model = d_model        
#        pe = torch.zeros(max_seq_len, d_model)
#        for pos in range(max_seq_len):
#            for i in range(0, d_model, 2):
#                pe[pos, i] = \
#                math.sin(pos / (10000 ** ((2 * i)/d_model)))
#                pe[pos, i + 1] = \
#                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
#        pe = pe.unsqueeze(0)
#        self.register_buffer('pe', pe)
# 
#    def forward(self, x):
#        # make embeddings relatively larger
#        x = x * math.sqrt(self.d_model)
#        #add constant to embedding
#        seq_len = x.size(1)
#        batch_size = x.size(0)
#        num_feature = x.size(2)
#        spatial_h = x.size(3)
#        spatial_w = x.size(4)
#        z = Variable(self.pe[:,:seq_len],requires_grad=False)
#        z = z.unsqueeze(-1).unsqueeze(-1)
#        z = z.expand(batch_size,seq_len, num_feature, spatial_h,  spatial_w)
#        x = x + z
#        return x
#
#
## standard attention layer
#def attention(q, k, v, d_k, mask=None, dropout=None):
#    scores = torch.sum(q * k , -1)/  math.sqrt(d_k)
#    # scores : b, t 
#    scores = F.softmax(scores, dim=-1)
#    scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
#    # scores : b, t, dim 
#    output = scores * v
#    output = torch.sum(output,1)
#    if dropout:
#        output = dropout(output)
#    return output
#
#
#
#
#class TX(nn.Module):
#    def __init__(self, d_model=64 , dropout = 0.3 ):
#        super(TX, self).__init__()
#        self.d_model = d_model
#        # no of head has been modified to encompass : 1024 dimension 
#        self.dropout = nn.Dropout(dropout)
#        self.dropout_2 = nn.Dropout(dropout)
#        self.norm_1 = Norm(d_model)
#        self.norm_2 = Norm(d_model)
#        self.ff = FeedForward(d_model, d_ff=d_model//2)
#    def forward(self, q, k, v, mask=None):
#        # q: (b , dim )
#        b = q.size(0)
#        t = k.size(1)
#        dim = q.size(1)
#        q_temp = q.unsqueeze(1)
#        q_temp= q_temp.expand(b, t , dim)
#        # q,k,v : (b, t , d_model=1024 // 16 )
#        A = attention(q_temp, k, v, self.d_model, mask, self.dropout)
#        # A : (b , d_model=1024 // 16 )
#        q_ = self.norm_1(A + q)
#        new_query = self.norm_2(q_ +  self.dropout_2(self.ff(q_))) 
#        return new_query
#
#
#class Block_head(nn.Module):
#    def __init__(self, d_model=64 , dropout = 0.3 ):
#        super(Block_head, self).__init__()
#        self.T1 = TX(d_model)
#        self.T2 = TX(d_model)
#        self.T3 = TX(d_model)
#    def forward(self, q, k, v, mask=None):
#        q = self.T1(q,k,v)
#        q = self.T2(q,k,v)
#        q = self.T3(q,k,v)
#        return q
#
#
#class Tail(nn.Module):
#    def __init__(self, num_classes , num_frames, head=16):
#        super(Tail, self).__init__()
#        self.spatial_h = 7
#        self.spatial_w = 7
#        self.head = head
#        self.num_features = 1024
#        self.num_frames = num_frames 
#        self.d_model = self.num_features // 2
#        self.d_k = self.d_model // self.head
#        self.bn1 = nn.BatchNorm2d(self.num_features)
#        self.bn2 = Norm(self.d_model, trainable=False)
#        
#        self.pos_embd = PositionalEncoder(self.num_features, self.num_frames)
#        self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(7,7), stride=1, padding=0, bias=False)
#
#        self.head_layers =[]
#        for i in range(self.head):
#            self.head_layers.append(Block_head(self.d_k))
#
#        self.list_layers = nn.ModuleList(self.head_layers)
#        self.classifier = nn.Linear(self.d_model, num_classes)
#        # resnet style initialization 
#        nn.init.kaiming_normal_(self.Qpr.weight, mode='fan_out')
#        nn.init.normal_(self.classifier.weight, std=0.001)  
#        # nn.init.constant(self.classifier.bias, 0)
#        
#        nn.init.constant_(self.bn1.weight , 1)
#        nn.init.constant_(self.bn1.bias , 0)
#        
#    def forward(self, x, b , t ):
#        x = self.bn1(x)
#        # stabilizes the learning
#        x = x.view(b , t , self.num_features , self.spatial_h , self.spatial_w)
#        x = self.pos_embd(x)
#        x = x.view(-1, self.num_features , self.spatial_h , self.spatial_w)
#        x = F.relu(self.Qpr(x))
#        # x: (b,t,1024,1,1) since its a convolution: spatial positional encoding is not added 
#        # paper has a different base (resnet in this case): which 2048 x 7 x 4 vs 16 x 7 x 7 
#        x = x.view(-1, t ,  self.d_model )
#        x = self.bn2(x)
#        # stabilization
#        q = x[:,t//2,:] #middle frame is the query
#        v = x # value
#        k = x #key 
#
#        q = q.view(b, self.head, self.d_k  )
#        k = k.view(b,t, self.head, self.d_k )
#        v = v.view(b,t, self.head, self.d_k )
#
#        k = k.transpose(1,2)
#        v = v.transpose(1,2)
#        #  q: b, 16, 64
#        #  k,v: b, 16, 10 ,64
#        outputs = []
#        for i in range(self.head):
#            outputs.append(self.list_layers[i](q[:,i],k[:,i], v[:,i]) )
#
#        f = torch.cat(outputs, 1)
#        f = F.normalize(f, p=2, dim=1)
#        # F.norma
##        if not self.training:
##            return f
#        y = self.classifier(f)
#        return y, f


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from libs.layers import PositionalEncoding
from ad_utils import get_const

class RNN_encoder(nn.Module):
    def __init__(self, dim_input, dim_lstm_hidden, reduce, bidirectional, dim_feature_mapping, nlayer):
        super(RNN_encoder, self).__init__()
        self.reduce = reduce
        self.dim_feature_mapping = dim_feature_mapping
        self.fm_layer = nn.Linear(dim_input, dim_feature_mapping)
        dim_lstm_input = dim_feature_mapping

        # encoder layer
        if bidirectional == 1:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=True, num_layers=nlayer)
            dim_att = dim_lstm_hidden * 2
        else:
            self.lstm_layer=nn.LSTM(input_size=dim_lstm_input, hidden_size=dim_lstm_hidden, bidirectional=False, num_layers=nlayer)
            dim_att = dim_lstm_hidden

        # readout
        if self.reduce == 'max' or self.reduce == 'mean':
            pass
        elif self.reduce == "self-attention":
            if bidirectional == 1:
                dim_att_in = 2 * dim_lstm_hidden
            elif bidirectional == 0:
                dim_att_in = dim_lstm_hidden
            else:
                print("bidirectional must be either 0 or 1")
                import sys; sys.exit(-1)

            self.att1 = nn.Linear(dim_att_in, dim_att)
            self.att2 = nn.Linear(dim_att, 1)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

    def forward(self, x):
        # x: (Bn, Tx, V, D), V is node-dimension
        Bn, Tx, V, D = x.size()
        
        x = x.view(Bn*Tx, V, D)

        x = torch.transpose(x, 0, 1).contiguous() # (V, Bn*Tx, D)

        x = self.fm_layer(x)

        # encoder
        ctx, hidden = self.lstm_layer(x, None) # ctx: (V, Bn*Tx, dim_hidden)

        # readout
        if self.reduce == "self-attention":
            att1 = torch.tanh(self.att1(ctx))
            att2 = self.att2(att1).view(V, Bn*Tx)

            alpha = att2 - torch.max(att2)
            alpha = torch.exp(alpha)

            alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
            enc_out = torch.sum(alpha.unsqueeze(2) * ctx, dim=0)
        elif self.reduce == "max":
            enc_out, _ = torch.max(ctx, dim=0)
        else:    
            enc_out = torch.mean(ctx, dim=0)

        enc_out = enc_out.view(Bn, Tx, -1)
        #print(enc_out)
        return enc_out # (Bn, Tx, D)

class Transformer_encoder(nn.Module):
    def __init__(self, dim_input, nhead, dim_feedforward, reduce, dim_feature_mapping, nlayer):
        super(Transformer_encoder, self).__init__()
        self.reduce=reduce
        self.dim_feature_mapping = dim_feature_mapping

        # use feature mapping
        self.fm_layer = nn.Linear(dim_input, dim_feature_mapping)
        d_model = self.dim_feature_mapping

        # self-attention
        if self.reduce == 'max' or self.reduce == 'mean':
            pass
        elif self.reduce == "self-attention":
            self.dim_att = d_model
            self.dim_att_in = d_model
            self.att1 = nn.Linear(self.dim_att_in, self.dim_att)
            self.att2 = nn.Linear(self.dim_att, 1)
        else:
            print("reduce must be either max, mean, or self-attention")
            import sys; sys.exit(-1)

        self.positionalEncoding = PositionalEncoding(d_model=d_model)
        self.t_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.t_layers = TransformerEncoder(encoder_layer=self.t_layer, num_layers=nlayer)

    def forward(self, x):
        Bn, Tx, V, D = x.size()
        
        x = x.view(Bn*Tx, V, D)

        x = torch.transpose(x, 0, 1).contiguous() # (V, Bn*Tx, D)

        x = self.fm_layer(x)

        x = self.positionalEncoding(x)
        ctx = self.t_layers(x)

        if self.reduce == "self-attention":
            att1 = torch.tanh(self.att1(ctx))
            att2 = self.att2(att1).view(V, Bn*Tx)

            alpha = att2 - torch.max(att2)
            alpha = torch.exp(alpha)

            alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
            enc_out = torch.sum(alpha.unsqueeze(2) * ctx, dim=0)
        elif self.reduce == "max":
            enc_out, _ = torch.max(ctx, dim=0)
        else:    
            enc_out = torch.mean(ctx, dim=0)

        enc_out = enc_out.view(Bn, Tx, -1)
        #print(enc_out)
        return enc_out # (Bn, Tx, D)

class DNN_classifier(nn.Module):
    def __init__(self, dim_input, n_fc_layers, dim_fc_hidden, drop_p, dim_output):
        super(DNN_classifier, self).__init__()

        fc_layers = []
        if n_fc_layers == 0:
            fc_layers += [nn.Linear(dim_input, dim_output)]
        else:
            fc_layers += [nn.Linear(dim_input, dim_fc_hidden), nn.ReLU(), nn.Dropout(p=drop_p)]
            for i in range(n_fc_layers-1):
               fc_layers +=[nn.Linear(dim_fc_hidden, dim_fc_hidden), nn.ReLU(), nn.Dropout(p=drop_p)]
            fc_layers += [nn.Linear(dim_fc_hidden, dim_output)]

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x): # x: (Bn x Tx x D)

        x = self.fc(x)

        return x
        #return F.log_softmax(x, dim=2), hidden

class RNN_classifier(nn.Module):
    def __init__(self, dim_input, n_lstm_layers, n_fc_layers, dim_lstm_hidden, dim_fc_hidden, drop_p, dim_output):
        super(RNN_classifier, self).__init__()

        fc_layers = []
        if n_fc_layers == 0:
            fc_layers += [nn.Linear(dim_lstm_hidden, dim_output)]
        else:
            fc_layers += [nn.Linear(dim_lstm_hidden, dim_fc_hidden), nn.ReLU(), nn.Dropout(p=drop_p)]
            for i in range(n_fc_layers-1):
               fc_layers +=[nn.Linear(dim_fc_hidden, dim_fc_hidden), nn.ReLU(), nn.Dropout(p=drop_p)]
            fc_layers += [nn.Linear(dim_fc_hidden, dim_output)]

        self.rnn = nn.LSTM(input_size=dim_input,
                           hidden_size=dim_lstm_hidden,
                           num_layers=n_lstm_layers)

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, hidden=None): # x: (Tx, Bn, D)

        x, hidden = self.rnn(x, hidden) # Tx, Bn, D

        x = self.fc(x)

        return x, hidden
        #return F.log_softmax(x, dim=2), hidden

class Transformer_enc_DNN_clf(nn.Module):
    def __init__(self, args):
        super(Transformer_enc_DNN_clf, self).__init__()

        self.encoder = Transformer_encoder(dim_input=args.dim_input, 
                                           nhead=args.nhead, 
                                           dim_feedforward=args.dim_feedforward,
                                           reduce=args.reduce,
                                           dim_feature_mapping=args.dim_feature_mapping,
                                           nlayer=args.nlayer)
        
        self.classifier = DNN_classifier(dim_input=args.dim_feature_mapping,
                                         n_fc_layers=args.clf_n_fc_layers,
                                         dim_fc_hidden=args.clf_dim_fc_hidden,
                                         drop_p=args.drop_p,
                                         dim_output=args.clf_dim_output)
    
    def forward(self, x, clf_hidden=None): # (Bn, V, D)
        # encoder
        x = self.encoder(x) # (Bn, Tx, dim_hidden) 

        x = x.squeeze()

        logits = self.classifier(x) # (V, 1, D)

        return logits

class Transformer_enc_RNN_clf(nn.Module):
    def __init__(self, args):
        super(Transformer_enc_RNN_clf, self).__init__()

        clf_dim_input = args.dim_feature_mapping
        self.clf_dim_lstm_hidden= args.clf_dim_lstm_hidden

        self.encoder = Transformer_encoder(dim_input=args.dim_input, 
                                           nhead=args.nhead, 
                                           dim_feedforward=args.dim_feedforward,
                                           reduce=args.reduce,
                                           dim_feature_mapping=args.dim_feature_mapping,
                                           nlayer=args.nlayer)
        
        self.classifier = RNN_classifier(dim_input=args.dim_feature_mapping,
                                         n_lstm_layers=args.clf_n_lstm_layers,
                                         n_fc_layers=args.clf_n_fc_layers,
                                         dim_lstm_hidden=args.clf_dim_lstm_hidden,
                                         dim_fc_hidden=args.clf_dim_fc_hidden,
                                         drop_p=args.drop_p,
                                         dim_output=args.clf_dim_output)

    def forward(self, x, clf_hidden=None):
        # encoder
        x = self.encoder(x) # (Bn, Tx, D) 
        x = torch.transpose(x, 0, 1).contiguous() # Tx, Bn, D
        
        logits, clf_hidden = self.classifier(x, clf_hidden) # (1, Bn, D)
        logits = logits[-1,:,:] # (1, dim_out)

        return logits, clf_hidden
    
    def init_clf_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.clf_dim_lstm_hidden, device=device), torch.zeros(1, batch_size, self.clf_dim_lstm_hidden, device=device))

class RNN_enc_RNN_clf(nn.Module): # RNN-enc + RNN classifier
    def __init__(self, args):
        super(RNN_enc_RNN_clf, self).__init__()

        if args.bidirectional==1:
            clf_dim_input=args.dim_lstm_hidden*2
        else:
            clf_dim_input=args.dim_lstm_hidden
        self.clf_dim_lstm_hidden= args.clf_dim_lstm_hidden

        # encoder
        self.encoder=RNN_encoder(dim_input = args.dim_input,
                                 dim_lstm_hidden = args.dim_lstm_hidden,
                                 reduce = args.reduce,
                                 bidirectional = args.bidirectional,
                                 dim_feature_mapping = args.dim_feature_mapping,
                                 nlayer = args.nlayer)

        # classifier
        self.classifier=RNN_classifier(dim_input = clf_dim_input,
                                       n_lstm_layers = args.clf_n_lstm_layers,
                                       n_fc_layers = args.clf_n_fc_layers,
                                       dim_lstm_hidden = args.clf_dim_lstm_hidden,
                                       dim_fc_hidden = args.clf_dim_fc_hidden,
                                       dim_output = args.clf_dim_output,
                                       drop_p=args.drop_p)

    def forward(self, x, clf_hidden=None): # (Bn, Tx, V, D)
        # encoder
        x = self.encoder(x) # (Bn, Tx, dim_hidden) 
        x = torch.transpose(x, 0, 1).contiguous() # Tx, Bn, dim_hidden

        logits, clf_hidden = self.classifier(x, clf_hidden) # (V, 1, D)
        logits = logits[-1,:,:] # (1, dim_out)

        return logits, clf_hidden
    
    def init_clf_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.clf_dim_lstm_hidden, device=device), torch.zeros(1, batch_size, self.clf_dim_lstm_hidden, device=device))

if __name__ == '__main__':
    mylayer = pooling_layer(reduce='mean')

    myvec = torch.tensor([[1,2,3,4],[1,5,2,3]]).type(torch.float32)

    layer_out = mylayer(myvec)

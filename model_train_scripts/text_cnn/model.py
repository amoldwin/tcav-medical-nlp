import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import residual

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
class CNN_Text_Plus_Residual(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text_Plus_Residual, self).__init__()
        self.args = args
       
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        final_cnn_dim = len(Ks) * Co
        #RMLP layers 
        if args.pretrained_residual==None:
            self.residual = nn.ModuleList([
                residual.ResidualMLP(args.num_structured_features, f=torch.nn.functional.tanh) for _ in range(args.num_res_layers)
            ])
        else:
            just_residual = WeightedResidualNetwork.from_pretrained(pretrained_residual, num_res_layers=10, num_structured_features=88)
            self.residual = just_residual.residual
        #second residual mlps for concatenated outputs
        self.final_residual = nn.ModuleList([
            residual.ResidualMLP(final_cnn_dim+args.num_structured_features, f=torch.nn.functional.tanh) for _ in range(args.num_res_layers)
        ])
        self.final_dense =  nn.Linear( final_cnn_dim+args.num_structured_features, final_cnn_dim+args.num_structured_features)
        self.fc1 = nn.Linear(final_cnn_dim+args.num_structured_features, C)
        
        
        
        
        
        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
#         self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x, structured_x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        
        x = self.dropout(x)  # (N, len(Ks)*Co)

        
        for l in self.residual:
            structured_x = l(structured_x)
        x = torch.cat((x, structured_x), dim=-1)
        for l in self.final_residual:
            x = l(x)
            
        x = self.final_dense(x)
        
        logits = self.fc1(x)

#         logit = self.fc1(x)  # (N, C)
        return logits

    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Tripletnet(nn.Module):
    def __init__(self, net, args):
        super(Tripletnet, self).__init__()
        self.net = net
        self.args = args
        """
		if False:
            self.metric_branch = nn.Linear(dim_embed, 300, bias=True)
            self.fc2 = nn.Linear(300, 1)
            # initilize as having an even weighting across all dimensions
		    weight = torch.zeros(1,dim_embed)/float(dim_embed)
            self.metric_branch.weight = nn.Parameter(weight)
		"""
    def forward(self, x, y, z=None):
        
        embedded_x = x
        embedded_y = y
        embedded_z = z
        if(z is None):
            dist_a = self.net(torch.cat((embedded_x, embedded_y), dim=1))
            return dist_a
        #dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        #dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        #print(torch.cat((embedded_x,embedded_y), dim=1).size())
        #print(torch.cat((embedded_x,embedded_z)).size())
        if(self.args.rand_cat):
            if(random.randint(0, 1)):
                dist_a = self.net(torch.cat((embedded_x, embedded_y), dim=1))
            else:
                dist_a = self.net(torch.cat((embedded_y, embedded_x), dim=1))
            if(random.randint(0, 1)):
                dist_b = self.net(torch.cat((embedded_x, embedded_z), dim=1))
            else:
                dist_b = self.net(torch.cat((embedded_z, embedded_x), dim=1))
        else:
            dist_a = self.net(torch.cat((embedded_x, embedded_y), dim=1))
            dist_b = self.net(torch.cat((embedded_x, embedded_z), dim=1))
        
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

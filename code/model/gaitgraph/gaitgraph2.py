import torch
import torch.nn as nn
from ..backbones.resgcn import ResGCN
from ..backbones.modules import Graph
import numpy as np


class GaitGraph2(nn.Module):
    """
        GaitGraph2: Towards a Deeper Understanding of Skeleton-based Gait Recognition
        Paper:    https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper
        Github:   https://github.com/tteepe/GaitGraph2
    """
    def __init__(self):
        super(GaitGraph2, self).__init__()
        self.joint_format = "coco"
        self.input_num = 3
        self.block = "Bottleneck"
        self.input_branch = [5,64,32]
        self.main_stream = [32,128,256]
        self.num_class = 155
        self.reduction = 8
        self.tta = "tta"
        ## Graph Init ##
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        ## Network ##
        self.ResGCN = ResGCN(input_num=self.input_num, input_branch=self.input_branch, 
                             main_stream=self.main_stream, num_class=self.num_class,
                             reduction=self.reduction, block=self.block,graph=self.A)
        self.mlp = nn.Sequential(
            nn.Linear(310,155),
            nn.Softmax(1)
            )
        

    def forward(self, inputs):

        x_input =inputs
        N, T, V, I, C = x_input.size()
        
        flip_idx = self.graph.flip_idx

        if self.tta:
            multi_input = MultiInput(self.graph.connect_joint, self.graph.center)
            x1 = []
            x2 = []
            for i in range(N):
                x1.append(multi_input(x_input[i,:,:,0,:3].flip(0)))
                x2.append(multi_input(x_input[i,:,flip_idx,0,:3]))
            x_input = torch.cat([ torch.stack(x1,0), torch.stack(x2,0)], dim=0)
        
        x = x_input.permute(0, 3, 4, 1, 2).contiguous()
        
        # resgcn
        x = self.ResGCN(x)

        if self.tta:
            f1, f2 = torch.split(x, [N, N], dim=0)
            x = torch.cat((f1, f2), dim=1)
             
        embed = self.mlp(x)
        # N, 154
        
        return embed
    
class MultiInput:
    def __init__(self, connect_joint, center):
        self.connect_joint = connect_joint
        self.center = center

    def __call__(self, data):

        # T, V, C -> T, V, I=3, C + 2
        T, V, C = data.shape
        x_new = torch.zeros((T, V, 3, C + 2), device=data.device)

        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]

        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]

        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        bone_length = 0
        for i in range(C - 1):
            bone_length += torch.pow(x_new[:, :, 2, i], 2)
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = torch.acos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]

        data = x_new
        return data


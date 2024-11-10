
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)
    
class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num=155, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()
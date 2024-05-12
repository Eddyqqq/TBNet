import torch
import torch.nn as nn
import torch.nn.functional as F

def gen_gt(sentence):
    lgt = len(sentence)
    gt = -torch.ones(lgt, lgt).cuda()
    for i, gls in enumerate(sentence):
        for k in range(lgt):
            if sentence[k] == gls:
                gt[i, k] = 1.0
    return gt

# class IteLoss(nn.Module):
#     def __init__(self):
#     def forward(self, gls_emd, vis_emd, label, label_lgt):


# class ItaLoss(nn.Module):
#     def __init__(self):
#     def forward(self, gls_emd, vis_emd, label, label_lgt):
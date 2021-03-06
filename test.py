from utils.utils import *
from pytorch_lightning.metrics import Precision, Recall, F1
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


t = torch.tensor([0,1,2,0,1,2])
p = torch.tensor([0,2,1,0,0,1])

pre = Precision(num_classes=3)
r = Recall(num_classes=3)

print(pre(p,t))
print(r(p,t))

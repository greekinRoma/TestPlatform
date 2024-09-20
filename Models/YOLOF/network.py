from torch import nn
import torch
class YOLOF(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

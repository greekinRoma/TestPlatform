import torch
from .DETR import DETR
from .YOLOF.network import YOLOF
from .DeFCN.network import DeFCN
def Get_Network(model_name:str,args,pth_path=None):
    if model_name == 'detr':
        return DETR(args)
    elif model_name == 'YOLOF':
        return YOLOF(args,pth=pth_path)
    elif model_name == 'DeFCN':
        return DeFCN(args,pth=pth_path)
    return None